#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"

namespace madevent {

class TwoParticle : public Mapping {
public:
    TwoParticle(bool _com) : Mapping(
        _com ?
        TypeList{scalar, scalar, scalar, scalar, scalar, scalar} :
        TypeList{scalar, scalar, scalar, scalar, scalar, scalar, four_vector},
        {four_vector, four_vector},
        {}
    ), com(_com) {}
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

private:
    bool com;
};


class TInvariantTwoParticle : public Mapping {
public:
    TInvariantTwoParticle(bool _com, double nu = 0, double mass = 0, double width = 0) :
        Mapping(
            {scalar, scalar, scalar, scalar},
            {four_vector, four_vector},
            {four_vector, four_vector}
        ), com(_com), invariant(nu, mass, width) {}
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

private:
    bool com;
    Invariant invariant;
};

}
