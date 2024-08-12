#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"


namespace madevent {

class Luminosity : public Mapping {
public:
    Luminosity(
        double _s_lab, double _s_hat_min, double _s_hat_max = 0,
        double nu = 0, double mass = 0, double width = 0
    ) :
        Mapping({scalar, scalar}, {scalar, scalar, scalar}, {}),
        s_lab(_s_lab),
        s_hat_min(_s_hat_min),
        s_hat_max(_s_hat_max == 0 ? _s_lab : _s_hat_max),
        invariant(nu, mass, width) {}
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

private:
    double s_lab, s_hat_min, s_hat_max;
    Invariant invariant;
};

}
