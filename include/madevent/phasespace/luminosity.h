#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/phasespace/invariants.h"


namespace madevent {

class Luminosity : public Mapping {
public:
    Luminosity(
        double _s_lab, double _s_hat_min, double _s_hat_max = 0,
        double nu = 1, double mass = 0, double width = 0
    ) :
        Mapping({batch_float, batch_float}, {batch_float, batch_float, batch_float}, {}),
        s_lab(_s_lab),
        s_hat_min(_s_hat_min),
        s_hat_max(_s_hat_max == 0 ? _s_lab : _s_hat_max),
        invariant(nu, mass, width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    double s_lab, s_hat_min, s_hat_max;
    Invariant invariant;
};

}
