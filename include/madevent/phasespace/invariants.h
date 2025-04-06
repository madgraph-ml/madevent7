#pragma once

#include "madevent/phasespace/mapping.h"


namespace madevent {

class Invariant : public Mapping {
public:
    Invariant(double _nu = 0, double _mass = 0, double _width = 0) :
        Mapping({batch_float}, {batch_float}, {batch_float, batch_float}),
        nu(_nu), mass(_mass), width(_width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    double nu, mass, width;
};

}
