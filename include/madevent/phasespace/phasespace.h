#pragma once

#include <vector>
#include <optional>

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/topology.h"
#include "madevent/phasespace/invariants.h"
#include "madevent/phasespace/two_particle.h"
#include "madevent/phasespace/luminosity.h"
#include "madevent/phasespace/t_propagator_mapping.h"
#include "madevent/phasespace/rambo.h"
#include "madevent/phasespace/cuts.h"
#include "madevent/phasespace/chili.h"

namespace madevent {

class PhaseSpaceMapping : public Mapping {
public:
    enum TChannelMode { propagator, rambo, chili };

    PhaseSpaceMapping(
        const Topology& topology,
        double s_lab,
        //double s_hat_min = 0.0,
        bool leptonic = false,
        double s_min_epsilon = 1e-2,
        double nu = 0.8,
        TChannelMode t_channel_mode = propagator,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::vector<Topology>& symmetric_topologies = {}
    );

    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double s_lab,
        bool leptonic = false,
        double s_min_epsilon = 1e-2,
        double nu = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt
    ) : PhaseSpaceMapping(
        Topology(
            [&] {
                if (external_masses.size() < 4) {
                    throw std::invalid_argument("The number of masses must be at least 4");
                }
                std::vector<Diagram::Vertex> vertices;
                auto n_out = external_masses.size() - 2;
                vertices.push_back({
                    {Diagram::incoming, 0},
                    {Diagram::propagator, 0},
                    {Diagram::outgoing, 0},
                });
                for (std::size_t i = 1; i < n_out - 1; ++i) {
                    vertices.push_back({
                        {Diagram::propagator, i - 1},
                        {Diagram::propagator, i},
                        {Diagram::outgoing, i},
                    });
                }
                vertices.push_back({
                    {Diagram::incoming, 1},
                    {Diagram::propagator, n_out - 2},
                    {Diagram::outgoing, n_out - 1},
                });
                return Diagram(
                    {external_masses.at(0), external_masses.at(1)},
                    {external_masses.begin() + 2, external_masses.end()},
                    std::vector<Propagator>(n_out - 1),
                    vertices
                );
            }(),
            Topology::no_decays
        ), s_lab, leptonic, s_min_epsilon, nu, mode, cuts
    ) {}
    std::size_t random_dim() const {
        return 3 * _outgoing_masses.size() - (_leptonic ? 4 : 2);
    }
    std::size_t particle_count() const {
        return _outgoing_masses.size() + 2;
    }
    std::size_t channel_count() const {
        return _permutations.size();
    }
private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    struct DecayMappings {
        std::size_t count;
        std::optional<Invariant> invariant;
        std::variant<TwoParticleDecay, FastRamboMapping, std::monostate> decay = std::monostate{};
    };

    double _pi_factors;
    double _s_lab;
    bool _leptonic;
    bool _has_t_channel;
    double _sqrt_s_epsilon;
    Cuts _cuts;
    std::vector<std::vector<DecayMappings>> _s_decays;
    std::variant<TPropagatorMapping, FastRamboMapping, ChiliMapping, std::monostate> _t_mapping;
    std::optional<Luminosity> _luminosity;
    std::vector<double> _outgoing_masses;
    std::vector<std::vector<std::size_t>> _permutations;
    std::vector<std::vector<std::size_t>> _inverse_permutations;
};

}
