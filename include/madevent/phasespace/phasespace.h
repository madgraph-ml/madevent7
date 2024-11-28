#pragma once

#include <vector>
#include <optional>

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/topology.h"
#include "madevent/phasespace/invariants.h"
#include "madevent/phasespace/two_particle.h"
#include "madevent/phasespace/luminosity.h"
#include "madevent/phasespace/t_propagator_mapping.h"


namespace madevent {

class PhaseSpaceMapping : public Mapping {
public:
    PhaseSpaceMapping(
        const Topology& topology,
        double s_lab,
        double s_hat_min = 0.0,
        bool leptonic = false,
        double s_min_epsilon = 1e-2,
        double nu = 1.4
    );
private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    struct DecayMappings {
        std::size_t count;
        std::optional<Invariant> invariant;
        std::optional<TwoParticle> decay;
    };

    double pi_factors;
    double s_lab;
    bool leptonic;
    bool has_t_channel;
    double sqrt_s_epsilon;
    std::vector<std::vector<DecayMappings>> s_decays;
    std::optional<TPropagatorMapping> t_mapping;
    std::optional<Luminosity> luminosity;
    std::vector<double> outgoing_masses;
    std::vector<std::size_t> permutation;
    std::vector<std::size_t> inverse_permutation;
};

}
