#pragma once

#include <vector>

#include "madevent/phasespace/topology.h"
#include "madevent/phasespace/base.h"
#include "madevent/phasespace/invariants.h"
#include "madevent/phasespace/two_particle.h"


namespace madevent {

class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(
        const std::vector<std::size_t>& integration_order,
        double nu=0.8
    );
    std::size_t random_dim() const {
        return 3 * _integration_order.size() - 1;
    }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    std::vector<std::size_t> _integration_order;
    std::vector<bool> _sample_sides;
    Invariant _uniform_invariant;
    TwoParticleScattering _com_scattering;
    TwoParticleScattering _lab_scattering;
};

}
