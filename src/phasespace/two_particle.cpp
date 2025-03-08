#include "madevent/phasespace/two_particle.h"

#include <cmath>

using namespace madevent;


Mapping::Result TwoParticle::build_forward_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto r1 = inputs.at(0), r2 = inputs.at(1);
    auto s = inputs.at(2), sqrt_s = inputs.at(3), m1 = inputs.at(4), m2 = inputs.at(5);
    auto p0 = com ? fb.com_momentum(sqrt_s) : inputs.at(6);
    auto phi = fb.uniform_phi(r1);
    auto costheta = fb.uniform_costheta(r2);
    auto [p1, gs] = fb.decay_momentum(s, sqrt_s, m1, m2);
    p1 = fb.rotate_zy(p1, phi, costheta);
    if (!com) p1 = fb.boost(p1, p0);
    auto p2 = fb.sub(p0, p1);
    return {{p1, p2}, gs};
}

Mapping::Result TwoParticle::build_inverse_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p0 = fb.add(p1, p2);
    auto [s, sqrt_s] = fb.s_and_sqrt_s(p0);
    auto m1 = fb.sqrt_s(p1);
    auto m2 = fb.sqrt_s(p2);
    if (!com) p1 = fb.boost_inverse(p1, p0);
    auto [phi, costheta] = fb.com_angles(p1);
    auto r1 = fb.uniform_phi_inverse(phi);
    auto r2 = fb.uniform_costheta_inverse(costheta);
    auto gs = fb.two_particle_density_inverse(s, m1, m2);
    if (com) {
        return {{r1, r2, s, sqrt_s, m1, m2}, gs};
    } else {
        return {{r1, r2, s, sqrt_s, m1, m2, p0}, gs};
    }
}

Mapping::Result TInvariantTwoParticle::build_forward_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto r1 = inputs.at(0), r2 = inputs.at(1), m1 = inputs.at(2), m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto p_tot = fb.add(p_in1, p_in2);
    auto [s, sqrt_s] = fb.s_and_sqrt_s(p_tot);
    auto s_in1 = fb.s(p_in1);
    auto s_in2 = fb.s(p_in2);
    auto p_in1_com = com ? p_in1 : fb.boost_inverse(p_in1, p_tot);
    auto [t_min, t_max] = fb.invt_min_max(s, s_in1, s_in2, m1, m2);
    auto [t_vec, det_t] = invariant.build_forward(fb, {r2}, {t_min, t_max});
    auto phi = fb.uniform_phi(r1);
    auto costheta = fb.invt_to_costheta(s, s_in1, s_in2, m1, m2, t_vec.at(0));
    auto [p1, _] = fb.decay_momentum(s, sqrt_s, m1, m2);
    p1 = fb.rotate_zy(p1, phi, costheta);
    if (!com) {
        auto [phi1, costheta1] = fb.com_angles(p_in1_com);
        p1 = fb.rotate_zy(p1, phi1, costheta1);
        p1 = fb.boost(p1, p_tot);
    }
    auto p2 = fb.sub(p_tot, p1);
    auto det = fb.tinv_two_particle_density(det_t, s, s_in1, s_in2);
    return {{p1, p2}, det};
}

Mapping::Result TInvariantTwoParticle::build_inverse_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto p_tot = fb.add(p_in1, p_in2);
    auto s = fb.s(p_tot);
    auto m1 = fb.sqrt_s(p1);
    auto m2 = fb.sqrt_s(p2);
    auto s_in1 = fb.s(p_in1);
    auto s_in2 = fb.s(p_in2);
    if (!com) {
        auto p_in1_com = fb.boost_inverse(p_in1, p_tot);
        auto [phi1, costheta1] = fb.com_angles(p_in1_com);
        p1 = fb.boost_inverse(p1, p_tot);
        p1 = fb.rotate_zy_inverse(p1, phi1, costheta1);
    }
    auto [phi, costheta] = fb.com_angles(p1);
    auto [t_min, t_max] = fb.invt_min_max(s, s_in1, s_in2, m1, m2);
    auto t = fb.costheta_to_invt(s, s_in1, s_in2, m1, m2, costheta);
    auto [r2_vec, det_t] = invariant.build_inverse(fb, {t}, {t_min, t_max});
    auto r1 = fb.uniform_phi_inverse(phi);
    auto det = fb.tinv_two_particle_density_inverse(det_t, s, s_in1, s_in2);
    return {{r1, r2_vec.at(0), m1, m2}, det};
}
