#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"

namespace madevent {

class TwoParticle : public Mapping {
public:
    TwoParticle(bool _com) : Mapping(
        [&] {
            TypeList input_types(6, batch_float);
            if (!_com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        {batch_four_vec, batch_four_vec},
        {}
    ), com(_com) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    bool com;
};


class TInvariantTwoParticle : public Mapping {
public:
    TInvariantTwoParticle(bool _com, double nu = 0, double mass = 0, double width = 0) :
        Mapping(
            {batch_float, batch_float, batch_float, batch_float},
            {batch_four_vec, batch_four_vec},
            {batch_four_vec, batch_four_vec}
        ), com(_com), invariant(nu, mass, width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    bool com;
    Invariant invariant;
};

}
