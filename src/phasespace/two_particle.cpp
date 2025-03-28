#include "madevent/phasespace/two_particle.h"

using namespace madevent;

Mapping::Result TwoParticle::build_forward_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto r_phi = inputs.at(0), r_cos_theta = inputs.at(1);
    auto m0 = inputs.at(2), m1 = inputs.at(3), m2 = inputs.at(4);
    auto [p1, p2, det] = _com ?
        fb.two_particle_decay_com(r_phi, r_cos_theta, m0, m1, m2) :
        fb.two_particle_decay(r_phi, r_cos_theta, m0, m1, m2, inputs.at(5));
    return {{p1, p2}, det};
}

Mapping::Result TwoParticle::build_inverse_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}

Mapping::Result TInvariantTwoParticle::build_forward_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2), m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_min, t_max] = fb.t_inv_min_max(p_in1, p_in2, m1, m2);
    auto [t_vec, det_inv] = _invariant.build_forward(fb, {r_inv}, {t_min, t_max});
    auto [p1, p2, det_scatter] = _com ?
        fb.two_particle_scattering_com(r_phi, p_in1, p_in2, t_vec.at(0), m1, m2) :
        fb.two_particle_scattering(r_phi, p_in1, p_in2, t_vec.at(0), m1, m2);
    return {{p1, p2}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result TInvariantTwoParticle::build_inverse_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
