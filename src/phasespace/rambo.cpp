#include "madevent/phasespace/rambo.h"

#include <cmath>

#include "madevent/constants.h"

using namespace madevent;

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
    Value r = fb.stack(ValueVec(inputs.begin(), inputs.begin() + random_dim()));
    Value e_cm = inputs.at(random_dim());

    std::array<Value, 2> output;
    if (massless) {
        if (com) {
            output = fb.fast_rambo_massless_com(r, e_cm);
        } else {
            Value p0 = inputs.back();
            output = fb.fast_rambo_massless(r, e_cm, p0);
        }
    } else {
        if (com) {
            ValueVec masses(inputs.begin() + random_dim(), inputs.end());
            output = fb.fast_rambo_massive_com(r, e_cm, fb.stack(masses));
        } else {
            ValueVec masses(inputs.begin() + random_dim(), inputs.end() - 1);
            Value p0 = inputs.back();
            output = fb.fast_rambo_massive(r, e_cm, fb.stack(masses), p0);
        }
    }
    return {fb.unstack(output[0]), output[1]};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
