#include "madevent/phasespace/integrand.h"

using namespace madevent;

ValueList DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const ValueList& args
) const {
    auto momenta = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);

    auto& [pids, me_index] = _pid_options.at(0);
    int64_t pid1 = pids.at(0);
    int64_t pid2 = pids.at(1);
    auto me2 = fb.matrix_element(momenta, static_cast<int64_t>(me_index));
    auto pdf1 = fb.pdf(x1, _q2, pid1);
    auto pdf2 = fb.pdf(x2, _q2, pid2);
    auto xs = fb.diff_cross_section(x1, x2, pdf1, pdf2, me2, _e_cm2);
    return {xs};
}

ValueList Integrand::build_function_impl(
    FunctionBuilder& fb, const ValueList& args
) const {
    auto r = _sample ?
        fb.random(args.at(0), static_cast<int64_t>(_mapping.random_dim())) :
        args.at(0);

    auto [momenta_x1_x2, det] = _mapping.build_forward(fb, {r}, {});
    auto momenta = momenta_x1_x2.at(0);
    auto x1 = momenta_x1_x2.at(1);
    auto x2 = momenta_x1_x2.at(2);

    auto indices = fb.nonzero(det);
    auto dxs = _diff_xs.build_function(
        fb, {fb.gather(indices, momenta), fb.gather(indices, x1), fb.gather(indices, x2)}
    );
    auto weights = fb.mul(fb.scatter(indices, det, dxs.at(0)), det);
    if (!_unweight) return {momenta, x1, x2, weights};

    auto [uw_indices, uw_weights] = fb.unweight(weights, args.at(1));
    return {
        fb.gather(uw_indices, momenta),
        fb.gather(uw_indices, x1),
        fb.gather(uw_indices, x2),
        uw_weights
    };
}
