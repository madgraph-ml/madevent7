#include "madevent/phasespace/three_particle.h"

using namespace madevent;

Mapping::Result ThreeBodyDecay::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_e1 = inputs.at(0), r_e2 = inputs.at(1);
    auto r_phi = inputs.at(2), r_cos_theta = inputs.at(3), r_beta = inputs.at(4);
    auto m0 = inputs.at(5), m1 = inputs.at(6), m2 = inputs.at(7), m3 = inputs.at(8);
    auto [p1, p2, p3, det] = _com ?
        fb.three_body_decay_com(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3) :
        fb.three_body_decay(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, inputs.at(9));
    return {{p1, p2, p3}, det};
}

Mapping::Result ThreeBodyDecay::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}

// Mapping::Result TwoToThreeParticleScattering::build_forward_impl(
//     FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
// ) const {
//     auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2), m2 = inputs.at(3);
//     auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
//     auto [t_min, t_max] = fb.t_inv_min_max(p_in1, p_in2, m1, m2);
//     auto [t_vec, det_inv] = _invariant.build_forward(fb, {r_inv}, {t_min, t_max});
//     auto [p1, p2, det_scatter] = fb.two_to_three_particle_scattering(r_phi, p_in1, p_in2, t_vec.at(0), m1, m2);
//     return {{p1, p2}, fb.mul(det_inv, det_scatter)};
// }

// Mapping::Result TwoToThreeParticleScattering::build_inverse_impl(
//     FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
// ) const {
//     throw std::logic_error("inverse mapping not implemented");
// }
