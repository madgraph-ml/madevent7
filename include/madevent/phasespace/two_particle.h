#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"

namespace madevent {

class TwoParticle : public Mapping {
public:
    TwoParticle(bool com) : Mapping(
        [&] {
            TypeVec input_types(5, batch_float);
            if (!com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        {batch_four_vec, batch_four_vec},
        {}
    ), _com(com) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;

    bool _com;
};


class TInvariantTwoParticle : public Mapping {
public:
    TInvariantTwoParticle(bool com, double nu = 0, double mass = 0, double width = 0) :
        Mapping(
            {batch_float, batch_float, batch_float, batch_float},
            {batch_four_vec, batch_four_vec},
            {batch_four_vec, batch_four_vec}
        ), _com(com), _invariant(nu, mass, width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;

    bool _com;
    Invariant _invariant;
};

}
