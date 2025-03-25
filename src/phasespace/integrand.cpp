#include "madevent/phasespace/integrand.h"

#include "madevent/util.h"

using namespace madevent;

ValueVec DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto momenta = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);

    auto& [pids, me_index] = _pid_options.at(0);
    int64_t pid1 = pids.at(0);
    int64_t pid2 = pids.at(1);
    auto pdf1 = fb.pdf(x1, _q2, pid1);
    auto pdf2 = fb.pdf(x2, _q2, pid2);
    if (_channel_count == 1) {
        auto me2 = fb.matrix_element(momenta, static_cast<int64_t>(me_index));
        auto xs = fb.diff_cross_section(x1, x2, pdf1, pdf2, me2, _e_cm2);
        return {xs};
    } else {
        auto [me2, chan_weights] = fb.matrix_element_multichannel(
            momenta, _amp2_remap, static_cast<int64_t>(me_index), _channel_count
        );
        auto xs = fb.diff_cross_section(x1, x2, pdf1, pdf2, me2, _e_cm2);
        return {xs, chan_weights};
    }
}

ValueVec Unweighter::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto [uw_indices, uw_weights] = fb.unweight(args.at(0), args.back());
    ValueVec output{uw_weights};
    for (auto arg : std::span(args.begin() + 1, args.end() - 1)) {
        output.push_back(fb.gather(uw_indices, arg));
    }
    return output;
}

ValueVec Integrand::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto r = _flags & sample ?
        fb.random(args.at(0), int64_t(_mapping.random_dim())) :
        args.at(0);

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

    auto [momenta_x1_x2, det] = _mapping.build_forward(fb, {r}, {});
    if (adaptive_det) {
        det = fb.mul(det, *adaptive_det);
    }
    auto momenta = momenta_x1_x2.at(0);
    auto x1 = momenta_x1_x2.at(1);
    auto x2 = momenta_x1_x2.at(2);

    auto indices = fb.nonzero(det);
    auto dxs = _diff_xs.build_function(
        fb, {fb.gather(indices, momenta), fb.gather(indices, x1), fb.gather(indices, x2)}
    );
    auto weights = fb.mul(fb.scatter(indices, det, dxs.at(0)), det);

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
