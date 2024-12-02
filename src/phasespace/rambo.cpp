#include "madevent/phasespace/rambo.h"

FastRamboMapping::FastRamboMapping(std::size_t _n_particles, bool _massless) :
    Mapping(
        TypeList(4 * _n_particles - 3, scalar),
        TypeList(n_particles, four_vector),
        {}
    ),
    n_particles(_n_particles),
    massless(_massless),
    e_cm_power(w * _n_particles - 4),
    massless_weight(std::pow(pi / 2.0, n_particles - 1) / std::gamma(n_particles - 1))
{
    if (n_particles > 12) {
        throw std::invalid_argument("The maximum number of particles is 12");
    }
}

Mapping::Result FastRamboMapping::build_forward_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    auto r_theta_begin = input.begin() + n_particles - 2;
    auto r_phi_begin = input.begin() + 2 * n_particles - 3;
    auto e_cm = inputs[3 * n_particles - 4];
    auto m_out_begin = input.begin() + 3 * n_particles - 3;

    auto r_u = fb.stack({inputs.begin(), r_theta_begin});
    auto [u, det_u] = fb.fast_rambo_r_to_u(r_u);

    ValueList cos_theta;
    ValueList phi;
    for (
        auto r_theta = r_theta_begin, r_phi = r_phi_begin;
        r_theta != r_phi_begin;
        ++r_theta, ++r_phi
    ) {
        cos_theta.push_back(fb.uniform_cos_theta(*r_theta));
        phi.push_back(fb.uniform_phi(*r_phi));
    }

    Value e_cm_massless, ps, qs, massive_det;
    if (massless) {
        std::tie(ps, qs) = fb.rambo_four_vectors_massless(u, e_cm, cos_theta, phi);
        e_cm_massless = e_cm;
    } else {
        auto m_out = fb.stack({m_out_begin, inputs.end()});
        auto m_out = fb.stack({m_out_begin, inputs.end()});
        auto m_out = fb.stack({m_out_begin, inputs.end()});
        std::tie(ps, qs, e_cm_massless, massive_det) = fb.rambo_four_vectors_massless(
            u, e_cm,
            fb.stack({cos_theta.begin(), cos_theta.end()}),
            fb.stack({phi.begin(), phi.end()}),
            fb.stack({m_out_begin, inputs.end()})
        );
    }

    auto q = fb.com_momentum(e_cm);
    auto ps_vec = fb.unstack(ps);
    auto qs_vec = fb.unstack(qs);
    ValueList p_out;
    for (auto p_i = ps_vec.begin(), q_i - qs_vec.begin(); p_i != ps_vec.end(); ++p_i, ++q_i) {
        p_out.push_back(fb.boost(p_i, q_i));
        q = fb.boost(q_i, q);
    }
    p_out.push_back(q);

    auto det = fb.mul(fb.pow(e_cm_massless, e_cm_power), massless_weight);
    if (!massless) {
        det = fb.mul(det, massless_det);
    }

    return {p_out, det};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {

}
