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
    const std::optional<SubchannelWeights>& subchan_weights,
    const std::optional<ChannelWeightNetwork>& chan_weight_net,
    const std::vector<me_int_t>& chan_weight_remap,
    std::size_t remapped_chan_count,
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
                    mapping.random_dim() + // phasespace
                    (mapping.channel_count() > 1) + // symmetric channel
                    (diff_xs.pid_options().size() > 1) + // flavor
                    diff_xs.has_mirror() // flipped initial state
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
                ret_types.push_back(batch_float_array(
                    chan_weight_remap.size() > 0 ? remapped_chan_count :
                    subchan_weights ? subchan_weights->channel_count() : diff_xs.channel_count()
                ));
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
    _subchan_weights(subchan_weights),
    _chan_weight_net(chan_weight_net),
    _chan_weight_remap(chan_weight_remap),
    _remapped_chan_count(remapped_chan_count),
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
    ChannelArgs channel_args {
        .r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0),
        .batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)}),
        .has_permutations = _mapping.channel_count() > 1,
        .has_multi_flavor = _diff_xs.pid_options().size() > 1,
        .has_mirror = _diff_xs.has_mirror(),
        .has_pdf_prior = _pdf1 && _energy_scale && channel_args.has_multi_flavor,
    };
    if (_flags & unweight) {
        channel_args.max_weight = args.at(1);
    }
    ChannelResult result = build_channel_part(fb, channel_args);
    return build_common_part(fb, channel_args, result);
}

Integrand::ChannelResult Integrand::build_channel_part(
    FunctionBuilder& fb, const Integrand::ChannelArgs& args
) const {
    ChannelResult result;
    ValueVec mapping_conditions;
    ValueVec weights_before_cuts, weights_after_cuts, adaptive_probs;

    // split up random numbers depending on discrete degrees of freedom
    Value chan_random, flavor_random, mirror_random;
    result.r = args.r;
    if (args.has_permutations) {
        auto [r_rest, r_val] = fb.pop(result.r);
        result.r = r_rest;
        chan_random = r_val;
    }
    if (args.has_multi_flavor) {
        auto [r_rest, r_val] = fb.pop(result.r);
        result.r = r_rest;
        flavor_random = r_val;
    }
    if (args.has_mirror) {
        auto [r_rest, r_val] = fb.pop(result.r);
        result.r = r_rest;
        mirror_random = r_val;
    }

    // if the channel contains multiple symmetry permutations, sample them either
    // uniformly, adaptively or NN-based depending on _discrete_before
    ValueVec flow_conditions;
    if (args.has_permutations) {
        me_int_t opt_count = _channel_indices.size();
        std::visit(Overloaded {
            [&](std::monostate) {
                auto [index, chan_det] = fb.sample_discrete(chan_random, opt_count);
                result.chan_index_in_group = index;
                weights_before_cuts.push_back(chan_det);
            },
            [&](auto discrete_before) {
                auto [index_vec, chan_det] = discrete_before.build_forward(
                    fb, {chan_random}, {}
                );
                result.chan_index_in_group = index_vec.at(0);
                weights_before_cuts.push_back(chan_det);
                adaptive_probs.push_back(chan_det);
            }
        }, _discrete_before);
        result.chan_index = fb.gather_int(result.chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(result.chan_index_in_group);
        //flow_conditions.push_back(fb.one_hot(chan_index_in_group, opt_count));
    }

    // apply VEGAS or MadNIS if given in _adaptive_map
    result.latent = result.r;
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
            auto [rs, r_det] = admap.build_forward(fb, {result.r}, cond);
            result.latent = rs.at(0);
            adaptive_probs.push_back(r_det);
            weights_before_cuts.push_back(r_det);
            flow_conditions.push_back(result.latent);
        }
    }, _adaptive_map);

    // apply phase space mapping
    auto [momenta_x1_x2, det] = _mapping.build_forward(
        fb, {result.latent}, mapping_conditions
    );
    weights_before_cuts.push_back(det);
    result.momenta = momenta_x1_x2.at(0);
    result.x1 = momenta_x1_x2.at(1);
    result.x2 = momenta_x1_x2.at(2);

    if (args.has_mirror) {
        auto [index, mirror_det] = fb.sample_discrete(
            mirror_random, static_cast<me_int_t>(2)
        );
        result.mirror_id = index;
        result.momenta_mirror = fb.mirror_momenta(result.momenta, result.mirror_id);
        weights_before_cuts.push_back(mirror_det);
    }

    // filter out events that did not pass cuts
    result.weight_before_cuts = fb.product(weights_before_cuts);
    result.indices_acc = fb.nonzero(result.weight_before_cuts);
    result.momenta_acc = fb.batch_gather(result.indices_acc, result.momenta);
    result.x1_acc = fb.batch_gather(result.indices_acc, result.x1);
    result.x2_acc = fb.batch_gather(result.indices_acc, result.x2);
    for (auto& cond : flow_conditions) {
        cond = fb.batch_gather(result.indices_acc, cond);
    }

    // if PDF grid and energy scale were given and the channel has more than one flavor,
    // evaluate energy scale and PDF for all flavors to use it as prior for flavor sampling
    if (args.has_pdf_prior) {
        auto scales = _energy_scale.value().build_function(fb, {result.momenta_acc});
        auto pdf1 = _pdf1.value().build_function(
            fb, {result.x1_acc, scales.at(1)}
        ).at(0);
        auto pdf2 = _pdf2.value().build_function(
            fb, {result.x2_acc, scales.at(2)}
        ).at(0);
        result.pdf_prior = fb.mul(
            fb.select(pdf1, _pdf_indices1), fb.select(pdf2, _pdf_indices2)
        );
        if (_active_flavors.size() > 0) {
            result.pdf_prior = fb.mul(result.pdf_prior, _active_flavors);
        }
        result.xs_cache = {pdf1, pdf2, scales.at(0)};
    }

    // if the channel has more than one flavor, sample them either uniformly,
    // adaptively or NN-based depending on _discrete_after
    result.flavor_id = static_cast<me_int_t>(0);
    if (args.has_multi_flavor) {
        auto flavor_random_acc = fb.batch_gather(result.indices_acc, flavor_random);
        std::visit(Overloaded {
            [&](std::monostate) {
                if (args.has_pdf_prior) {
                    auto [index, flavor_det] = fb.sample_discrete_probs(
                        flavor_random_acc, result.pdf_prior
                    );
                    result.flavor_id = index;
                    weights_after_cuts.push_back(flavor_det);
                } else {
                    auto [index, flavor_det] = fb.sample_discrete(
                        flavor_random_acc,
                        static_cast<me_int_t>(_diff_xs.pid_options().size())
                    );
                    result.flavor_id = index;
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
                if (args.has_pdf_prior) {
                    discrete_condition.push_back(result.pdf_prior);
                }
                auto [index_vec, flavor_det] = discrete_after.build_forward(
                    fb, {flavor_random_acc}, discrete_condition
                );
                result.flavor_id = index_vec.at(0);
                weights_after_cuts.push_back(flavor_det);
                auto ones = fb.full({1., args.batch_size});
                adaptive_probs.push_back(fb.batch_scatter(
                    result.indices_acc, ones, flavor_det
                ));
            }
        }, _discrete_after);
    }
    result.weight_after_cuts = fb.product(weights_after_cuts);
    result.adaptive_prob = fb.product(adaptive_probs);
    return result;
}

ValueVec Integrand::build_common_part(
    FunctionBuilder& fb, const Integrand::ChannelArgs& args, const Integrand::ChannelResult& result
) const {
    // if _prop_chan_weights is given, compute channel weight based
    // on denominators of propagators
    Value chan_weights_acc;
    std::size_t channel_count =
        _chan_weight_remap.size() > 0 ? _remapped_chan_count :
        _subchan_weights ? _subchan_weights->channel_count() : _diff_xs.channel_count();
    if (channel_count > 1 && _prop_chan_weights) {
        chan_weights_acc = _prop_chan_weights->build_function(
            fb, {result.momenta_acc}
        ).at(0);
    }

    // evaluate differential cross section
    ValueVec xs_args {result.momenta_acc, result.x1_acc, result.x2_acc, result.flavor_id};
    if (args.has_mirror) xs_args.push_back(result.mirror_id);
    xs_args.insert(xs_args.end(), result.xs_cache.begin(), result.xs_cache.end());
    auto dxs_vec = _diff_xs.build_function(fb, xs_args);
    auto diff_xs_acc = dxs_vec.at(0);
    ValueVec weights_after_cuts{result.weight_after_cuts, diff_xs_acc};
    if (!_prop_chan_weights) {
        chan_weights_acc = dxs_vec.at(1);
    }
    if (channel_count > 1 && _subchan_weights) {
        chan_weights_acc = _subchan_weights->build_function(
            fb, {result.momenta_acc, chan_weights_acc}
        ).at(0);
    }
    if (_chan_weight_remap.size() > 0) {
        chan_weights_acc = fb.collect_channel_weights(
            chan_weights_acc, _chan_weight_remap, _remapped_chan_count
        );
    }

    // if given, apply channel weight network
    auto prior_chan_weights_acc = chan_weights_acc;
    if (_chan_weight_net) {
        chan_weights_acc = _chan_weight_net.value().build_function(
            fb, {result.momenta_acc, result.x1_acc, result.x2_acc, chan_weights_acc}
        ).at(0);
    }

    // compute full phase-space weight
    Value selected_chan_weight_acc;
    if (channel_count > 1) {
        //TODO fixme
        Value chan_index_acc;
        if (args.has_permutations) {
            chan_index_acc = fb.batch_gather(result.indices_acc, result.chan_index);
        } else {
            chan_index_acc = static_cast<me_int_t>(_channel_indices.at(0));
        }
        selected_chan_weight_acc = fb.gather(chan_index_acc, chan_weights_acc);
        weights_after_cuts.push_back(selected_chan_weight_acc);
    } else {
        selected_chan_weight_acc = 1.;
    }
    auto weight = fb.mul(result.weight_before_cuts, fb.batch_scatter(
        result.indices_acc, result.weight_before_cuts, fb.product(weights_after_cuts)
    ));

    // return results based on _flags
    ValueVec outputs{weight};
    if (_flags & return_momenta) {
        outputs.push_back(args.has_mirror ? result.momenta_mirror : result.momenta);
    }
    if (_flags & return_x1_x2) {
        outputs.push_back(result.x1);
        outputs.push_back(result.x2);
    }
    if (_flags & return_random) {
        outputs.push_back(result.r);
    }
    if (_flags & return_latent) {
        outputs.push_back(result.latent);
        outputs.push_back(result.adaptive_prob);
    }
    if (_flags & return_channel) {
        Value ret_chan_index = args.has_permutations ?
            result.chan_index :
            fb.full(
                {static_cast<me_int_t>(_channel_indices.at(0)), args.batch_size}
            );
        outputs.push_back(ret_chan_index);
    }
    if (_flags & return_chan_weights) {
        auto cw_flat = fb.full({
            1. / channel_count, args.batch_size, static_cast<me_int_t>(channel_count)
        });
        auto ones = fb.full({1., args.batch_size});
        if (channel_count > 1) {
            outputs.push_back(
                fb.batch_scatter(result.indices_acc, cw_flat, prior_chan_weights_acc)
            );
            outputs.push_back(
                fb.batch_scatter(result.indices_acc, ones, selected_chan_weight_acc)
            );
        } else {
            outputs.push_back(cw_flat);
            outputs.push_back(ones);
        }
    }
    if (_flags & return_cwnet_input) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc = preproc.build_function(
            fb, {result.momenta_acc, result.x1_acc, result.x2_acc}
        ).at(0);
        auto zeros = fb.full(
            {0., args.batch_size, static_cast<me_int_t>(preproc.output_dim())}
        );
        outputs.push_back(fb.batch_scatter(result.indices_acc, zeros, cw_preproc_acc));
    }
    if (_flags & return_discrete) {
        if (
            args.has_permutations &&
            !std::holds_alternative<std::monostate>(_discrete_before)
        ) {
            outputs.push_back(result.chan_index_in_group);
        }
        if (
            args.has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)
        ) {
            auto zeros = fb.full({static_cast<me_int_t>(0), args.batch_size});
            outputs.push_back(
                fb.batch_scatter(result.indices_acc, zeros, result.flavor_id)
            );
        }
    }
    if (_flags & return_discrete_latent) {
        Value chan_index_in_group = result.chan_index_in_group;
        if (!args.has_permutations) {
            chan_index_in_group = fb.full(
                {static_cast<me_int_t>(_channel_indices.at(0)), args.batch_size}
            );
        }
        outputs.push_back(chan_index_in_group);
        if (
            args.has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)
        ) {
            auto zeros = fb.full({static_cast<me_int_t>(0), args.batch_size});
            outputs.push_back(
                fb.batch_scatter(result.indices_acc, zeros, result.flavor_id)
            );
            if (args.has_pdf_prior) {
                auto flav_count = _diff_xs.pid_options().size();
                auto norm = fb.full({
                    1. / flav_count, args.batch_size, static_cast<me_int_t>(flav_count)
                });
                outputs.push_back(
                    fb.batch_scatter(result.indices_acc, norm, result.pdf_prior)
                );
            }
        }
    }

    if (_flags & unweight) {
        Unweighter unweighter(return_types());
        ValueVec unweighter_args = outputs;
        unweighter_args.push_back(args.max_weight);
        outputs = unweighter.build_function(fb, unweighter_args);
    }

    return outputs;

}

MultiChannelIntegrand::MultiChannelIntegrand(
    const std::vector<std::shared_ptr<Integrand>>& integrands
) :
    FunctionGenerator(
        "MultiChannelIntegrand",
        [&] {
            TypeVec arg_types;
            for (auto& arg_type : integrands.at(0)->arg_types()) {
                if (arg_type.dtype == DataType::batch_sizes) {
                    if (arg_type.batch_size_list.size() != 1) {
                        throw std::invalid_argument(
                            "Only batch size list arguments with size 1 accepted"
                        );
                    }
                } else {
                    arg_types.push_back(arg_type);
                }
            }
            arg_types.push_back(multichannel_batch_size(integrands.size()));
            return arg_types;
        }(),
        integrands.at(0)->return_types()
    ),
    _integrands(integrands)
{
    auto& first_function = integrands.at(0);
    std::size_t arg_count = first_function->arg_types().size();
    std::size_t return_count = first_function->return_types().size();
    for (auto& integrand : integrands) {
        if (
            integrand->arg_types().size() != arg_count ||
            integrand->return_types().size() != return_count
        ) {
            throw std::invalid_argument(
                "All integrands must have the same number of inputs and outputs"
            );
        }
    }
}

ValueVec MultiChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    /*Integrand::ChannelArgs common_args {
        .r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0),
        .batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)}),
        .has_permutations = _mapping.channel_count() > 1,
        .has_multi_flavor = _diff_xs.pid_options().size() > 1,
        .has_mirror = _diff_xs.has_mirror(),
        .has_pdf_prior = _pdf1 && _energy_scale && channel_args.has_multi_flavor,
    };
    if (_flags & unweight) {
        channel_args.max_weight = args.at(1);
    }*/
    return {};
    //ChannelResult result = build_channel_part(fb, channel_args);
    //return build_common_part(fb, channel_args, result);
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
            fb.one_hot(chan_index, static_cast<me_int_t>(_permutation_count))
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
