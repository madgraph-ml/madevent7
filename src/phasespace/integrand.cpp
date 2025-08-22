#include "madevent/phasespace/integrand.h"

#include "madevent/util.h"

#include <set>

using namespace madevent;

Unweighter::Unweighter(const TypeVec& types) :
    FunctionGenerator(
        "Unweighter",
        [&] {
            auto arg_types = types;
            arg_types.push_back(single_float);
            return arg_types;
        }(),
        types
    )
{}

ValueVec Unweighter::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto [uw_indices, uw_weights] = fb.unweight(args.at(0), args.back());
    ValueVec output{uw_weights};
    for (auto arg : std::span(args.begin() + 1, args.end() - 1)) {
        output.push_back(fb.batch_gather(uw_indices, arg));
    }
    return output;
}

Integrand::Integrand(
    const PhaseSpaceMapping& mapping,
    const DifferentialCrossSection& diff_xs,
    const AdaptiveMapping& adaptive_map,
    const AdaptiveDiscrete& discrete_before,
    const AdaptiveDiscrete& discrete_after,
    const std::optional<PdfGrid>& pdf_grid,
    const std::optional<EnergyScale>& energy_scale,
    const std::optional<PropagatorChannelWeights>& prop_chan_weights,
    const std::optional<ChannelWeightNetwork>& chan_weight_net,
    int flags,
    const std::vector<std::size_t>& channel_indices,
    const std::vector<std::size_t>& active_flavors
) :
    FunctionGenerator(
        "Integrand",
        [&] {
            TypeVec arg_types;
            if (flags & sample) {
                arg_types.push_back(Type({batch_size}));
            } else {
                arg_types.push_back(batch_float_array(
                    mapping.random_dim() + (diff_xs.channel_count() > 1)
                ));
            }
            if (flags & unweight) arg_types.push_back(single_float);
            return arg_types;
        }(),
        [&] {
            TypeVec ret_types {batch_float};
            if (flags & return_momenta) {
                ret_types.push_back(batch_four_vec_array(mapping.particle_count()));
            }
            if (flags & return_x1_x2) {
                ret_types.push_back(batch_float);
                ret_types.push_back(batch_float);
            }
            if (flags & return_random) {
                ret_types.push_back(batch_float_array(mapping.random_dim()));
            }
            if (flags & return_latent) {
                if (std::holds_alternative<std::monostate>(adaptive_map)) {
                    throw std::invalid_argument(
                        "return_latent flag can only be set if adaptive mapping is present"
                    );
                }
                ret_types.push_back(batch_float_array(mapping.random_dim()));
                ret_types.push_back(batch_float);
            }
            if (flags & return_channel) {
                ret_types.push_back(batch_int);
            }
            if (flags & return_chan_weights) {
                ret_types.push_back(batch_float_array(diff_xs.channel_count()));
                ret_types.push_back(batch_float);
            }
            if (flags & return_cwnet_input) {
                ret_types.push_back(batch_float_array(
                    chan_weight_net.value().preprocessing().output_dim()
                ));
            }
            auto flav_count = diff_xs.pid_options().size();
            if (flags & return_discrete) {
                if (
                    mapping.channel_count() > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_before)
                ) {
                    ret_types.push_back(batch_int);
                }
                if (
                    flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)
                ) {
                    ret_types.push_back(batch_int);
                }
            }
            if (flags & return_discrete_latent) {
                ret_types.push_back(batch_int);
                if (
                    flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)
                ) {
                    ret_types.push_back(batch_int);
                    if (pdf_grid && energy_scale) {
                        ret_types.push_back(batch_float_array(flav_count));
                    }
                }
            }
            return ret_types;
        }()
    ),
    _mapping(mapping),
    _diff_xs(diff_xs),
    _adaptive_map(adaptive_map),
    _discrete_before(discrete_before),
    _discrete_after(discrete_after),
    _energy_scale(energy_scale),
    _prop_chan_weights(prop_chan_weights),
    _chan_weight_net(chan_weight_net),
    _flags(flags),
    _channel_indices(channel_indices.begin(), channel_indices.end()),
    _random_dim(
        mapping.random_dim() + // phasespace
        (mapping.channel_count() > 1) + // symmetric channel
        (diff_xs.pid_options().size() > 1) + // flavor
        diff_xs.has_mirror() // flipped initial state
    )
{
    if (pdf_grid) {
        std::set<int> pids1, pids2;
        for (auto& option : diff_xs.pid_options()) {
            pids1.insert(option.at(0));
            pids2.insert(option.at(1));
        }
        for (auto& option : diff_xs.pid_options()) {
            _pdf_indices1.push_back(std::distance(pids1.begin(), pids1.find(option.at(0))));
            _pdf_indices2.push_back(std::distance(pids2.begin(), pids2.find(option.at(1))));
        }
        _pdf1 = PartonDensity(pdf_grid.value(), {pids1.begin(), pids1.end()});
        _pdf2 = PartonDensity(pdf_grid.value(), {pids2.begin(), pids2.end()});
        if (active_flavors.size() > 0) {
            _active_flavors.resize(diff_xs.pid_options().size());
            for (auto index : active_flavors) {
                _active_flavors.at(index) = 1.;
            }
        }
    }
}

std::tuple<std::vector<std::size_t>, std::vector<bool>> Integrand::latent_dims() const {
    std::vector<std::size_t> dims {_mapping.random_dim(), 1};
    std::vector<bool> is_float {true, false};

    auto flav_count = _diff_xs.pid_options().size();
    if (
        flav_count > 1 &&
        !std::holds_alternative<std::monostate>(_discrete_after)
    ) {
        dims.push_back(1);
        is_float.push_back(false);
        if (_pdf1 && _energy_scale) {
            dims.push_back(flav_count);
            is_float.push_back(true);
        }
    }

    return {dims, is_float};
}

ValueVec Integrand::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    Value r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0);
    Value batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)});
    ValueVec mapping_conditions;
    ValueVec weights_before_cuts, weights_after_cuts, adaptive_probs;

    // split up random numbers depending on discrete degrees of freedom
    Value chan_random, flavor_random, mirror_random;
    bool has_permutations = _mapping.channel_count() > 1;
    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
    bool has_mirror = _diff_xs.has_mirror();
    if (has_permutations) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        chan_random = r_val;
    }
    if (has_multi_flavor) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        flavor_random = r_val;
    }
    if (has_mirror) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        mirror_random = r_val;
    }

    // if the channel contains multiple symmetry permutations, sample them either
    // uniformly, adaptively or NN-based depending on _discrete_before
    Value chan_index, chan_index_in_group;
    ValueVec flow_conditions;
    if (has_permutations) {
        int64_t opt_count = _channel_indices.size();
        std::visit(Overloaded {
            [&](std::monostate) {
                auto [index, chan_det] = fb.sample_discrete(chan_random, opt_count);
                chan_index_in_group = index;
                weights_before_cuts.push_back(chan_det);
            },
            [&](auto discrete_before) {
                auto [index_vec, chan_det] = discrete_before.build_forward(
                    fb, {chan_random}, {}
                );
                chan_index_in_group = index_vec.at(0);
                weights_before_cuts.push_back(chan_det);
                adaptive_probs.push_back(chan_det);
            }
        }, _discrete_before);
        chan_index = fb.gather_int(chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(chan_index_in_group);
        //flow_conditions.push_back(fb.one_hot(chan_index_in_group, opt_count));
    }

    // apply VEGAS or MadNIS if given in _adaptive_map
    auto latent = r;
    std::visit(Overloaded {
        [&](std::monostate) {},
        [&](auto& admap) {
            ValueVec cond;
            using TAdaptive = std::decay_t<decltype(admap)>;
            if constexpr (std::is_same_v<TAdaptive, Flow>) {
                if (flow_conditions.size() == 1) {
                    cond.push_back(flow_conditions.at(0));
                } else if (flow_conditions.size() > 1) {
                    cond.push_back(fb.cat(flow_conditions));
                }
            }
            auto [rs, r_det] = admap.build_forward(fb, {r}, cond);
            latent = rs.at(0);
            adaptive_probs.push_back(r_det);
            weights_before_cuts.push_back(r_det);
            flow_conditions.push_back(latent);
        }
    }, _adaptive_map);

    // apply phase space mapping
    auto [momenta_x1_x2, det] = _mapping.build_forward(fb, {latent}, mapping_conditions);
    weights_before_cuts.push_back(det);
    auto momenta = momenta_x1_x2.at(0);
    auto x1 = momenta_x1_x2.at(1);
    auto x2 = momenta_x1_x2.at(2);

    Value mirror_id, momenta_mirror;
    if (has_mirror) {
        auto [index, mirror_det] = fb.sample_discrete(
            mirror_random, static_cast<int64_t>(2)
        );
        mirror_id = index;
        momenta_mirror = fb.mirror_momenta(momenta, mirror_id);
        weights_before_cuts.push_back(mirror_det);
    }

    // filter out events that did not pass cuts
    auto weight = fb.product(weights_before_cuts);
    auto indices_acc = fb.nonzero(weight);
    auto momenta_acc = fb.batch_gather(indices_acc, momenta);
    auto x1_acc = fb.batch_gather(indices_acc, x1);
    auto x2_acc = fb.batch_gather(indices_acc, x2);
    for (auto& cond : flow_conditions) {
        cond = fb.batch_gather(indices_acc, cond);
    }

    // if _prop_chan_weights is given, compute channel weight based
    // on denominators of propagators
    Value chan_weights_acc;
    bool has_multi_channel = _diff_xs.channel_count() > 1;
    if (_prop_chan_weights && has_multi_channel) {
        chan_weights_acc = _prop_chan_weights.value().build_function(
            fb, {momenta_acc}
        ).at(0);
    }

    // if PDF grid and energy scale were given and the channel has more than one flavor,
    // evaluate energy scale and PDF for all flavors to use it as prior for flavor sampling
    bool has_pdf_prior = _pdf1 && _energy_scale && has_multi_flavor;
    ValueVec xs_cache;
    Value pdf_prior;
    if (has_pdf_prior) {
        auto scales = _energy_scale.value().build_function(fb, {momenta_acc});
        auto pdf1 = _pdf1.value().build_function(
            fb, {x1_acc, scales.at(1)}
        ).at(0);
        auto pdf2 = _pdf2.value().build_function(
            fb, {x2_acc, scales.at(2)}
        ).at(0);
        pdf_prior = fb.mul(
            fb.select(pdf1, _pdf_indices1), fb.select(pdf2, _pdf_indices2)
        );
        if (_active_flavors.size() > 0) {
            pdf_prior = fb.mul(pdf_prior, _active_flavors);
        }
        xs_cache = {pdf1, pdf2, scales.at(0)};
    }

    // if the channel has more than one flavor, sample them either uniformly,
    // adaptively or NN-based depending on _discrete_after
    Value flavor_id(static_cast<int64_t>(0));
    if (has_multi_flavor) {
        auto flavor_random_acc = fb.batch_gather(indices_acc, flavor_random);
        std::visit(Overloaded {
            [&](std::monostate) {
                if (has_pdf_prior) {
                    auto [index, flavor_det] = fb.sample_discrete_probs(
                        flavor_random_acc, pdf_prior
                    );
                    flavor_id = index;
                    weights_after_cuts.push_back(flavor_det);
                } else {
                    auto [index, flavor_det] = fb.sample_discrete(
                        flavor_random_acc,
                        static_cast<int64_t>(_diff_xs.pid_options().size())
                    );
                    flavor_id = index;
                    weights_after_cuts.push_back(flavor_det);
                }
            },
            [&](auto discrete_after) {
                ValueVec discrete_condition;
                using TDiscrete = std::decay_t<decltype(discrete_after)>;
                if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                    if (flow_conditions.size() == 1) {
                        discrete_condition.push_back(flow_conditions.at(0));
                    } else if (flow_conditions.size() > 1) {
                        discrete_condition.push_back(fb.cat(flow_conditions));
                    }
                }
                if (has_pdf_prior) {
                    discrete_condition.push_back(pdf_prior);
                }
                auto [index_vec, flavor_det] = discrete_after.build_forward(
                    fb, {flavor_random_acc}, discrete_condition
                );
                flavor_id = index_vec.at(0);
                weights_after_cuts.push_back(flavor_det);
                auto ones = fb.full({1., batch_size});
                adaptive_probs.push_back(fb.batch_scatter(indices_acc, ones, flavor_det));
            }
        }, _discrete_after);
    }

    // evaluate differential cross section
    ValueVec xs_args {momenta_acc, x1_acc, x2_acc, flavor_id};
    if (has_mirror) xs_args.push_back(mirror_id);
    xs_args.insert(xs_args.end(), xs_cache.begin(), xs_cache.end());
    auto dxs_vec = _diff_xs.build_function(fb, xs_args);
    auto diff_xs_acc = dxs_vec.at(0);
    weights_after_cuts.push_back(diff_xs_acc);
    if (!_prop_chan_weights) {
        chan_weights_acc = dxs_vec.at(1);
    }

    // if given, apply channel weight network
    auto prior_chan_weights_acc = chan_weights_acc;
    if (_chan_weight_net) {
        chan_weights_acc = _chan_weight_net.value().build_function(
            fb, {momenta_acc, x1_acc, x2_acc, chan_weights_acc}
        ).at(0);
    }

    // compute full phase-space weight
    Value selected_chan_weight_acc;
    if (has_multi_channel) {
        //TODO fixme
        Value chan_index_acc;
        if (has_permutations) {
            chan_index_acc = fb.batch_gather(indices_acc, chan_index);
        } else {
            chan_index_acc = static_cast<int64_t>(_channel_indices.at(0));
        }
        selected_chan_weight_acc = fb.gather(chan_index_acc, chan_weights_acc);
        weights_after_cuts.push_back(selected_chan_weight_acc);
    } else {
        selected_chan_weight_acc = 1.;
    }
    weight = fb.mul(weight, fb.batch_scatter(indices_acc, weight, fb.product(weights_after_cuts)));

    // return results based on _flags
    ValueVec outputs{weight};
    if (_flags & return_momenta) {
        outputs.push_back(has_mirror ? momenta_mirror : momenta);
    }
    if (_flags & return_x1_x2) {
        outputs.push_back(x1);
        outputs.push_back(x2);
    }
    if (_flags & return_random) {
        outputs.push_back(r);
    }
    if (_flags & return_latent) {
        outputs.push_back(latent);
        outputs.push_back(fb.product(adaptive_probs));
    }
    if (_flags & return_channel) {
        if (!has_permutations) {
            chan_index = fb.full({static_cast<int64_t>(_channel_indices.at(0)), batch_size});
        }
        outputs.push_back(chan_index);
    }
    if (_flags & return_chan_weights) {
        auto chan_count = _diff_xs.channel_count();
        auto cw_flat = fb.full({
            1. / chan_count, batch_size, static_cast<int64_t>(chan_count)
        });
        auto ones = fb.full({1., batch_size});
        if (has_multi_channel) {
            outputs.push_back(fb.batch_scatter(indices_acc, cw_flat, prior_chan_weights_acc));
            outputs.push_back(fb.batch_scatter(indices_acc, ones, selected_chan_weight_acc));
        } else {
            outputs.push_back(cw_flat);
            outputs.push_back(ones);
        }
    }
    if (_flags & return_cwnet_input) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc = preproc.build_function(
            fb, {momenta_acc, x1_acc, x2_acc}
        ).at(0);
        auto zeros = fb.full({0., batch_size, static_cast<int64_t>(preproc.output_dim())});
        outputs.push_back(fb.batch_scatter(indices_acc, zeros, cw_preproc_acc));
    }
    if (_flags & return_discrete) {
        if (has_permutations && !std::holds_alternative<std::monostate>(_discrete_before)) {
            outputs.push_back(chan_index_in_group);
        }
        if (has_multi_flavor && !std::holds_alternative<std::monostate>(_discrete_after)) {
            auto zeros = fb.full({static_cast<int64_t>(0), batch_size});
            outputs.push_back(fb.batch_scatter(indices_acc, zeros, flavor_id));
        }
    }
    if (_flags & return_discrete_latent) {
        if (!has_permutations) {
            chan_index_in_group = fb.full({static_cast<int64_t>(_channel_indices.at(0)), batch_size});
        }
        outputs.push_back(chan_index_in_group);
        if (
            has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)
        ) {
            auto zeros = fb.full({static_cast<int64_t>(0), batch_size});
            outputs.push_back(fb.batch_scatter(indices_acc, zeros, flavor_id));
            if (has_pdf_prior) {
                auto flav_count = _diff_xs.pid_options().size();
                auto norm = fb.full({
                    1. / flav_count, batch_size, static_cast<int64_t>(flav_count)
                });
                outputs.push_back(fb.batch_scatter(indices_acc, norm, pdf_prior));
            }
        }
    }

    if (_flags & unweight) {
        Unweighter unweighter(return_types());
        ValueVec unweighter_args = outputs;
        unweighter_args.push_back(args.at(1));
        outputs = unweighter.build_function(fb, unweighter_args);
    }

    return outputs;
}

IntegrandProbability::IntegrandProbability(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandProbability",
        [&] {
            TypeVec arg_types {
                batch_float_array(integrand._mapping.random_dim()),
                batch_int
            };
            auto flavor_count = integrand._diff_xs.pid_options().size();
            if (
                flavor_count > 1 &&
                !std::holds_alternative<std::monostate>(integrand._discrete_after)
            ) {
                arg_types.push_back(batch_int);
                if (integrand._pdf1 && integrand._energy_scale) {
                    arg_types.push_back(batch_float_array(flavor_count));
                }
            }
            return arg_types;

        }(),
        {batch_float}
    ),
    _adaptive_map(integrand._adaptive_map),
    _discrete_before(integrand._discrete_before),
    _discrete_after(integrand._discrete_after),
    _permutation_count(integrand._mapping.channel_count()),
    _flavor_count(integrand._diff_xs.pid_options().size()),
    _has_pdf_prior(integrand._pdf1 && integrand._energy_scale)
{}

ValueVec IntegrandProbability::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    ValueVec probs, flow_conditions;
    if (_permutation_count > 1) {
        auto chan_index = args.at(1);
        /*flow_conditions.push_back(
            fb.one_hot(chan_index, static_cast<int64_t>(_permutation_count))
        );*/
        std::visit(Overloaded {
            [](std::monostate) {},
            [&](auto discrete_before) {
                auto [r, chan_det] = discrete_before.build_inverse(fb, {chan_index}, {});
                probs.push_back(chan_det);
            }
        }, _discrete_before);
    }

    auto latent = args.at(0);
    std::visit(Overloaded {
        [&](std::monostate) {},
        [&](auto& admap) {
            ValueVec cond;
            using TAdaptive = std::decay_t<decltype(admap)>;
            if constexpr (std::is_same_v<TAdaptive, Flow>) {
                if (flow_conditions.size() == 1) {
                    cond.push_back(flow_conditions.at(0));
                } else if (flow_conditions.size() > 1) {
                    cond.push_back(fb.cat(flow_conditions));
                }
            }
            auto [r, r_det] = admap.build_inverse(fb, {latent}, cond);
            probs.push_back(r_det);
            flow_conditions.push_back(latent);
        }
    }, _adaptive_map);

    std::size_t arg_index = 2;
    if (_flavor_count > 1) {
        std::visit(Overloaded {
            [&](std::monostate) {},
            [&](auto discrete_after) {
                auto flavor = args.at(arg_index);
                ++arg_index;
                ValueVec discrete_condition;
                using TDiscrete = std::decay_t<decltype(discrete_after)>;
                if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                    if (flow_conditions.size() == 1) {
                        discrete_condition.push_back(flow_conditions.at(0));
                    } else if (flow_conditions.size() > 1) {
                        discrete_condition.push_back(fb.cat(flow_conditions));
                    }
                }
                if (_has_pdf_prior) {
                    auto pdf_prior = args.at(arg_index);
                    ++arg_index;
                    discrete_condition.push_back(pdf_prior);
                }
                auto [r_flavor, flavor_det] = discrete_after.build_inverse(
                    fb, {flavor}, discrete_condition
                );
                probs.push_back(flavor_det);
            }
        }, _discrete_after);
    }

    return {fb.product(probs)};
}
