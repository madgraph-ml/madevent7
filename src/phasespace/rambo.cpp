#include "madevent/phasespace/rambo.h"

#include <ranges>
#include <cmath>

#include "madevent/constants.h"

using namespace madevent;
namespace views = std::views;
namespace ranges = std::ranges;

FastRamboMapping::FastRamboMapping(std::size_t _n_particles, bool _massless, bool _com) :
    Mapping(
        [&] {
            TypeList input_types(3 * _n_particles - 3 + (_massless ? 0 : _n_particles), batch_float);
            if (!_com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        TypeList(_n_particles, batch_four_vec),
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
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    auto r_u = inputs | views::take(n_particles - 2)
                      | ranges::to<ValueList>();
    auto cos_theta = inputs | views::drop(n_particles - 2)
                            | views::take(n_particles - 1)
                            | views::transform([&fb](auto r) { return fb.uniform_costheta(r); })
                            | ranges::to<ValueList>();
    auto phi = inputs | views::drop(2 * n_particles - 3)
                      | views::take(n_particles - 1)
                      | views::transform([&fb](auto r) { return fb.uniform_phi(r); })
                      | ranges::to<ValueList>();
    auto e_cm = inputs.at(3 * n_particles - 4);
    auto m_out = inputs | views::drop(3 * n_particles - 3)
                        | views::take(n_particles)
                        | ranges::to<ValueList>();

    auto [u, det_u] = fb.fast_rambo_r_to_u(fb.stack(r_u));

    Value e_cm_massless, ps, qs, massive_det;
    if (massless) {
        std::tie(ps, qs) = fb.rambo_four_vectors_massless(
            u, e_cm, fb.stack(cos_theta), fb.stack(phi)
        );
        e_cm_massless = e_cm;
    } else {
        std::tie(ps, qs, e_cm_massless, massive_det) = fb.rambo_four_vectors_massive(
            u, e_cm, fb.stack(cos_theta), fb.stack(phi), fb.stack(m_out)
        );
    }

    auto q = com ? fb.com_momentum(e_cm) : inputs.back();
    ValueList p_out;
    for (auto [p_i, q_i] : views::zip(fb.unstack(ps), fb.unstack(qs))) {
        p_out.push_back(fb.boost(p_i, q));
        q = fb.boost(q_i, q);
    }
    p_out.push_back(q);

    auto det = fb.product({fb.pow(e_cm_massless, e_cm_power), massless_weight, det_u});
    if (!massless) {
        det = fb.mul(det, massive_det);
    }

    return {p_out, det};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {

}
