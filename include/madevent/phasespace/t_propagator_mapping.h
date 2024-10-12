#pragma once

#include <vector>

#include "madevent/phasespace/topology.h"
#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"
#include "madevent/phasespace/two_particle.h"


namespace madevent {

class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(
        const std::vector<Propagator>& propagators,
        double nu=0.,
        bool map_resonances=false
    );
    std::size_t random_dim() {
        return 4 * t_invariants.size() + 1;
    }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    std::vector<TInvariantTwoParticle> t_invariants;
    std::vector<Invariant> s_pseudo_invariants;
};

}
