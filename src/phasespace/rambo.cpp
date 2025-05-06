#include "madevent/phasespace/rambo.h"

#include <ranges>
#include <cmath>

#include "madevent/constants.h"
#include "madevent/util.h"

using namespace madevent;
namespace views = std::views;
namespace ranges = std::ranges;

FastRamboMapping::FastRamboMapping(std::size_t _n_particles, bool _massless, bool _com) :
    Mapping(
        [&] {
            TypeVec input_types(3 * _n_particles - 3 + (_massless ? 0 : _n_particles), batch_float);
            if (!_com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        TypeVec(_n_particles, batch_four_vec),
        {}
    ),
    n_particles(_n_particles),
    massless(_massless),
    com(_com),
    e_cm_power(2 * _n_particles - 4),
    massless_weight(std::pow(PI / 2.0, _n_particles - 1) / std::tgamma(_n_particles - 1))
{
    if (n_particles < 3 || n_particles > 12) {
        throw std::invalid_argument("The number of particles must be between 3 and 12");
    }
}

Mapping::Result FastRamboMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    ValueVec r_u, cos_theta, phi, m_out;
    auto it = inputs.begin();
    for (; it != inputs.begin() + n_particles - 2; ++it) {
        r_u.push_back(*it);
    }
    for (; it != inputs.begin() + 2 * n_particles - 3; ++it) {
        cos_theta.push_back(fb.uniform_costheta(*it));
    }
    for (; it != inputs.begin() + 3 * n_particles - 4; ++it) {
        phi.push_back(fb.uniform_phi(*it));
    }
    Value e_cm = *(it++);
    for (; it != inputs.end(); ++it) {
        m_out.push_back(*it);
    }

    auto [u, det_u] = fb.fast_rambo_r_to_u(fb.stack(r_u));

    std::array<Value, 4> rambo_four_vectors;
    if (massless) {
        auto [ps, qs] = fb.rambo_four_vectors_massless(
            u, e_cm, fb.stack(cos_theta), fb.stack(phi)
        );
        rambo_four_vectors = {ps, qs, e_cm, Value{}};
    } else {
        rambo_four_vectors = fb.rambo_four_vectors_massive(
            u, e_cm, fb.stack(cos_theta), fb.stack(phi), fb.stack(m_out)
        );
    }
    auto [ps, qs, e_cm_massless, massive_det] = rambo_four_vectors;

    auto q = com ? fb.com_momentum(e_cm) : inputs.back();
    ValueVec p_out;
    for (auto [p_i, q_i] : zip(fb.unstack(ps), fb.unstack(qs))) {
        p_out.push_back(fb.boost(p_i, q));
        q = fb.boost(q_i, q);
    }
    p_out.push_back(q);

    auto det = fb.product(fb.stack(
        {fb.pow(e_cm_massless, e_cm_power), massless_weight, det_u}
    ));
    if (!massless) {
        det = fb.mul(det, massive_det);
    }

    return {p_out, det};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
