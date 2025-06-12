#include "madevent/phasespace/integrand.h"

#include "madevent/util.h"

#include <set>

using namespace madevent;

Unweighter::Unweighter(const TypeVec& types) :
    FunctionGenerator(
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
    const std::vector<std::size_t>& channel_indices
) :
    FunctionGenerator(
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
                ret_types.push_back(batch_float_array(mapping.random_dim()));
            }
            if (flags & return_discrete) {
                if (mapping.channel_count() > 1) {
                    ret_types.push_back(batch_int);
                }
                auto flav_count = diff_xs.pid_options().size();
                if (flav_count > 1) {
                    ret_types.push_back(batch_int);
                    if (pdf_grid && energy_scale) {
                        ret_types.push_back(batch_float_array(flav_count));
                    }
                }
            }
            if (flags & return_chan_weights) {
                ret_types.push_back(batch_float_array(diff_xs.channel_count()));
            }
            if (flags & return_cwnet_input) {
                ret_types.push_back(batch_float_array(
                    chan_weight_net.value().preprocessing().output_dim()
                ));
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
        (diff_xs.pid_options().size() > 1) // flavor
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
    }
}


ValueVec Integrand::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    Value r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0);
    ValueVec mapping_conditions;
    ValueVec weights_before_cuts, weights_after_cuts;

    // split up random numbers depending on discrete degrees of freedom
    Value chan_random, flavor_random;
    bool has_permutations = _mapping.channel_count() > 1;
    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
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

    // if the channel contains multiple symmetry permutations, sample them either
    // uniformly, adaptively or NN-based depending on _discrete_before
    Value chan_index, chan_index_in_group;
    if (has_permutations) {
        std::visit(Overloaded {
            [&](std::monostate) {
                auto [index, chan_det] = fb.sample_discrete(
                    chan_random, static_cast<int64_t>(_channel_indices.size())
                );
                chan_index_in_group = index;
                weights_before_cuts.push_back(chan_det);
            },
            [&](auto discrete_before) {
                auto [index_vec, chan_det] = discrete_before.build_forward(
                    fb, {chan_random}, {}
                );
                chan_index_in_group = index_vec.at(0);
                weights_before_cuts.push_back(chan_det);
            }
        }, _discrete_before);
        chan_index = fb.gather_int(chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(chan_index_in_group);
    }

    // apply VEGAS or MadNIS if given in _adaptive_map
    auto latent = r;
    std::visit(Overloaded {
        [&](std::monostate) {},
        [&](auto& admap) {
            //TODO: make conditional on permutation index
            auto [rs, r_det] = admap.build_forward(fb, {r}, {});
            latent = rs.at(0);
            weights_before_cuts.push_back(r_det);
        }
    }, _adaptive_map);

    // apply phase space mapping
    auto [momenta_x1_x2, det] = _mapping.build_forward(fb, {latent}, mapping_conditions);
    weights_before_cuts.push_back(det);
    auto momenta = momenta_x1_x2.at(0);
    auto x1 = momenta_x1_x2.at(1);
    auto x2 = momenta_x1_x2.at(2);

    // filter out events that did not pass cuts
    auto weight = fb.product(weights_before_cuts);
    auto indices_acc = fb.nonzero(weight);
    auto momenta_acc = fb.batch_gather(indices_acc, momenta);
    auto x1_acc = fb.batch_gather(indices_acc, x1);
    auto x2_acc = fb.batch_gather(indices_acc, x2);

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
        auto scales = _energy_scale.value().build_function(fb, {momenta});
        auto pdf1 = _pdf1.value().build_function(fb, {x1, scales.at(1)}).at(0);
        auto pdf2 = _pdf2.value().build_function(fb, {x2, scales.at(2)}).at(0);
        pdf_prior = fb.mul(
            fb.select(pdf1, _pdf_indices1), fb.select(pdf2, _pdf_indices2)
        );
        xs_cache = {pdf1, pdf2, scales.at(0)};
    }

    // if the channel has more than one flavor, sample them either uniformly,
    // adaptively or NN-based depending on _discrete_after
    Value flavor_id(static_cast<int64_t>(0)), mirror_id(static_cast<int64_t>(0));
    if (has_multi_flavor) {
        std::visit(Overloaded {
            [&](std::monostate) {
                if (has_pdf_prior) {
                    auto [index, flavor_det] = fb.sample_discrete_probs(
                        flavor_random, pdf_prior
                    );
                    flavor_id = index;
                    weights_after_cuts.push_back(flavor_det);
                } else {
                    auto [index, flavor_det] = fb.sample_discrete(
                        flavor_random, static_cast<int64_t>(_diff_xs.pid_options().size())
                    );
                    flavor_id = index;
                    weights_after_cuts.push_back(flavor_det);
                }
            },
            [&](auto discrete_after) {
                //TODO: make conditional on previous things
                ValueVec discrete_condition;
                if (has_pdf_prior) {
                    discrete_condition.push_back(pdf_prior);
                }
                auto [index_vec, flavor_det] = discrete_after.build_forward(
                    fb, {flavor_random}, discrete_condition
                );
                flavor_id = index_vec.at(0);
                weights_after_cuts.push_back(flavor_det);
            }
        }, _discrete_after);
    }

    // evaluate differential cross section
    ValueVec xs_args {momenta_acc, x1_acc, x2_acc, flavor_id, mirror_id};
    xs_args.insert(xs_args.end(), xs_cache.begin(), xs_cache.end());
    auto dxs_vec = _diff_xs.build_function(fb, xs_args);
    auto diff_xs_acc = dxs_vec.at(0);
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
    if (has_multi_channel) {
        auto chan_index_acc = fb.batch_gather(indices_acc, chan_index);
        weights_after_cuts.push_back(fb.gather(chan_index_acc, chan_weights_acc));
    }
    weight = fb.mul(fb.scatter(indices_acc, weight, fb.product(weights_after_cuts)), det);

    // return results based on _flags
    ValueVec outputs{weight};
    if (_flags & return_momenta) {
        outputs.push_back(momenta);
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
    }
    if (_flags & return_discrete) {
        if (has_permutations && !std::holds_alternative<std::monostate>(_discrete_after)) {
            outputs.push_back(chan_index_in_group);
        }
        if (has_multi_flavor && !std::holds_alternative<std::monostate>(_discrete_before)) {
            Value batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)});
            auto zeros = fb.full({static_cast<int64_t>(0), batch_size});
            outputs.push_back(fb.scatter(indices_acc, zeros, flavor_id));
            if (has_pdf_prior) {
                auto flav_count = _diff_xs.pid_options().size();
                auto norm = fb.full({
                    1. / flav_count, batch_size, static_cast<int64_t>(flav_count)
                });
                outputs.push_back(pdf_prior);
            }
        }
    }
    if (_flags & return_chan_weights) {
        Value batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)});
        auto chan_count = _diff_xs.channel_count();
        auto cw_flat = fb.full({
            1. / chan_count, batch_size, static_cast<int64_t>(chan_count)
        });
        outputs.push_back(fb.scatter(indices_acc, cw_flat, prior_chan_weights_acc));
    }
    if (_flags & return_cwnet_input) {
        Value batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)});
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc = preproc.build_function(
            fb, {momenta_acc, x1_acc, x2_acc}
        ).at(0);
        auto zeros = fb.full({0., batch_size, static_cast<int64_t>(preproc.output_dim())});
        outputs.push_back(fb.scatter(indices_acc, cw_preproc_acc, zeros));
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
        [&] {
            TypeVec arg_types;
            if (integrand._mapping.channel_count() > 1) {
                arg_types.push_back(batch_int);
            }
            arg_types.push_back(batch_float_array(integrand._mapping.random_dim()));
            auto flavor_count = integrand._diff_xs.pid_options().size();
            if (flavor_count > 1) {
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
    ValueVec probs;
    std::size_t arg_index = 0;

    if (_permutation_count > 1) {
        auto chan_index = args.at(arg_index);
        ++arg_index;
        std::visit(Overloaded {
            [&](std::monostate) {
                probs.push_back(1. / _permutation_count);
            },
            [&](auto discrete_before) {
                auto [r, chan_det] = discrete_before.build_inverse(fb, {chan_index}, {});
                probs.push_back(chan_det);
            }
        }, _discrete_before);
    }

    auto latent = args.at(arg_index);
    ++arg_index;
    std::visit(Overloaded {
        [&](std::monostate) {},
        [&](auto& admap) {
            //TODO: make conditional on permutation index
            auto [r, r_det] = admap.build_inverse(fb, {latent}, {});
            probs.push_back(r_det);
        }
    }, _adaptive_map);

    if (_flavor_count > 1) {
        auto flavor = args.at(arg_index);
        ++arg_index;
        std::visit(Overloaded {
            [&](std::monostate) {
                if (_has_pdf_prior) {
                    auto pdf_prior = args.at(arg_index);
                    ++arg_index;
                    probs.push_back(fb.gather(flavor, pdf_prior));
                } else {
                    probs.push_back(1. / _flavor_count);
                }
            },
            [&](auto discrete_after) {
                //TODO: make conditional on previous things
                ValueVec discrete_condition;
                if (_has_pdf_prior) {
                    auto pdf_prior = args.at(arg_index);
                    ++arg_index;
                    discrete_condition.push_back(pdf_prior);
                }
                auto [r_flavor, flavor_det] = discrete_after.build_forward(
                    fb, {flavor}, discrete_condition
                );
                probs.push_back(flavor_det);
            }
        }, _discrete_after);
    }

    return {fb.product(probs)};
}
