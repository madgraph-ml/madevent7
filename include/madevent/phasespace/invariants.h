#pragma once

#include "madevent/phasespace/mapping.h"


namespace madevent {

class Invariant : public Mapping {
public:
    Invariant(double _nu = 0, double _mass = 0, double _width = 0) :
        Mapping({scalar}, {scalar}, {scalar, scalar}),
        nu(_nu), mass(_mass), width(_width) {}
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

private:
    double nu, mass, width;
};

}
