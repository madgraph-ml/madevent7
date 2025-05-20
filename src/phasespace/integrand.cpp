#include "madevent/phasespace/integrand.h"

#include "madevent/util.h"

using namespace madevent;

DifferentialCrossSection::DifferentialCrossSection(
    const std::vector<std::vector<int64_t>>& pid_options,
    std::size_t matrix_element_index,
    double e_cm2,
    double q2,
    bool simple_matrix_element,
    std::size_t channel_count,
    const std::vector<int64_t>& amp2_remap
) :
    FunctionGenerator(
        [&] {
            TypeVec arg_types {
                batch_four_vec_array(pid_options.at(0).size()),
                batch_float,
                batch_float,
                batch_int,
                batch_int
            };
            if (!simple_matrix_element) {
                arg_types.push_back(batch_float);
            }
            return arg_types;
        }(),
        simple_matrix_element ?
            TypeVec{batch_float} :
            TypeVec{batch_float_array(channel_count), batch_int, batch_int}
    ),
    _pid_options(pid_options),
    _simple_matrix_element(simple_matrix_element),
    _matrix_element_index(matrix_element_index),
    _e_cm2(e_cm2),
    _q2(q2),
    _channel_count(channel_count),
    _amp2_remap(amp2_remap)
{}

ValueVec DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto momenta = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);
    auto flavor_id = args.at(3);
    auto mirror_id = args.at(4);

    auto& pids = _pid_options.at(0);
    Value pid1, pid2;
    if (_pid_options.size() > 1) {
        std::vector<int64_t> pid1_options;
        std::vector<int64_t> pid2_options;
        for (auto& option : _pid_options) {
            pid1_options.push_back(option.at(0));
            pid2_options.push_back(option.at(1));
        }
        pid1 = fb.batch_gather(flavor_id, pid1_options);
        pid2 = fb.batch_gather(flavor_id, pid2_options);
    } else {
        pid1 = pids.at(0);
        pid2 = pids.at(1);
    }
    auto pdf1 = fb.pdf(x1, _q2, pid1);
    auto pdf2 = fb.pdf(x2, _q2, pid2);
    if (_simple_matrix_element) {
        auto me2 = fb.matrix_element(momenta, flavor_id, mirror_id, _matrix_element_index);
        auto xs = fb.diff_cross_section(x1, x2, pdf1, pdf2, me2, _e_cm2);
        return {xs};
    } else {
        auto alpha_s = args.at(5);
        auto batch_size = fb.batch_size({momenta, alpha_s, flavor_id, mirror_id});
        auto random = fb.random(batch_size, 2LL);
        auto [me2, chan_weights, color_id, diagram_id] = fb.matrix_element_multichannel(
            momenta, alpha_s, random, flavor_id, mirror_id,
            _amp2_remap, _matrix_element_index, _channel_count
        );
        auto xs = fb.diff_cross_section(x1, x2, pdf1, pdf2, me2, _e_cm2);
        return {xs, chan_weights, color_id, diagram_id};
    }
}

Unweighter::Unweighter(const TypeVec& types, std::size_t particle_count) :
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
            return ret_types;
        }()
    ),
    _mapping(mapping),
    _diff_xs(diff_xs),
    _adaptive_map(adaptive_map),
    _flags(flags),
    _channel_indices(channel_indices.begin(), channel_indices.end()),
    _random_dim(mapping.random_dim() + (diff_xs.channel_count() > 1))
{}


ValueVec Integrand::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    Value r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0);
    Value chan_index, chan_det;
    ValueVec mapping_conditions;
    if (_diff_xs.channel_count() > 1) {
        auto [r_new, chan_random] = fb.pop(r);
        auto [chan_index_in_group, chan_det_new] = fb.sample_discrete(
            chan_random, static_cast<int64_t>(_channel_indices.size())
        );
        r = r_new;
        chan_det = chan_det_new;
        chan_index = fb.gather_int(chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(chan_index_in_group);
    }

    std::optional<Value> adaptive_det;
    std::visit(
        Overloaded {
            [&](std::monostate) {},
            [&](auto& admap) {
                auto [rs, r_det] = admap.build_forward(fb, {r}, {});
                r = rs.at(0);
                adaptive_det = r_det;
            }
        },
        _adaptive_map
    );

    auto [momenta_x1_x2, det] = _mapping.build_forward(fb, {r}, mapping_conditions);
    if (adaptive_det) {
        det = fb.mul(det, *adaptive_det);
    }
    auto momenta = momenta_x1_x2.at(0);
    auto x1 = momenta_x1_x2.at(1);
    auto x2 = momenta_x1_x2.at(2);

    auto indices = fb.nonzero(det);
    auto dxs = _diff_xs.build_function(
        fb,
        {
            fb.batch_gather(indices, momenta),
            fb.batch_gather(indices, x1),
            fb.batch_gather(indices, x2)
        }
    );
    auto acc_weights = dxs.at(0);
    if (_diff_xs.channel_count() > 1) {
        auto chan_weights = dxs.at(1);
        auto acc_chan_index = fb.batch_gather(indices, chan_index);
        auto acc_chan_det = fb.batch_gather(indices, chan_det);
        acc_weights = fb.mul(
            fb.mul(acc_weights, acc_chan_det),
            fb.gather(acc_chan_index, chan_weights)
        );
    }
    auto weights = fb.mul(fb.scatter(indices, det, acc_weights), det);
    //auto weights = det; //fb.mul(acc_weights, det);

    ValueVec outputs{weights};
    if (_flags & return_momenta) outputs.push_back(momenta);
    if (_flags & return_x1_x2) {
        outputs.push_back(x1);
        outputs.push_back(x2);
    }
    if (_flags & return_random) outputs.push_back(r);

    if (_flags & unweight) {
        Unweighter unweighter(return_types(), _mapping.particle_count());
        ValueVec unweighter_args = outputs;
        unweighter_args.push_back(args.at(1));
        outputs = unweighter.build_function(fb, unweighter_args);
    }

    return outputs;
}
