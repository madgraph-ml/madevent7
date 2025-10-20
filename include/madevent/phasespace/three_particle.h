#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/phasespace/invariants.h"

namespace madevent {

class ThreeBodyDecay : public Mapping {
public:
    ThreeBodyDecay(bool com) : Mapping(
        "ThreeBodyDecay",
        [&] {
            TypeVec input_types(9, batch_float);
            if (!com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        {batch_four_vec, batch_four_vec, batch_four_vec},
        {}
    ), _com(com) {}
    std::size_t random_dim() const { return 5; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    bool _com;
};


// class TwoToThreeParticleScattering : public Mapping {
// public:
//     TwoToThreeParticleScattering(
//         bool com, double invariant_power = 0, double mass = 0, double width = 0
//     ) :
//         Mapping(
//             "TwoToThreeParticleScattering",
//             {batch_float, batch_float, batch_float, batch_float},
//             {batch_four_vec, batch_four_vec},
//             {batch_four_vec, batch_four_vec}
//         ), _com(com), _invariant(invariant_power, mass, width) {}

// private:
//     Result build_forward_impl(
//         FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
//     ) const override;
//     Result build_inverse_impl(
//         FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
//     ) const override;

//     bool _com;
//     Invariant _invariant;
// };

}
