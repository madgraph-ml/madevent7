#pragma once

#include <vector>

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/invariants.h"


namespace madevent {


class Diagram {
public:
    friend class TPropagatorMapping;
    friend class DiagramMapping;

    using IndexList = std::vector<std::size_t>;
    struct Propagator {
        double mass;
        double width;
    };
    struct Decay {
        std::optional<Propagator&> propagator;
        std::size_t count;
    }

    Diagram(
        std::vector<double> incoming_masses,
        std::vector<double> outgoing_masses,
        std::vector<Propagator> propagators,
        std::vector<IndexList> vertices
    ) :
        incoming_masses(_incoming_masses), outgoing_masses(_outgoing_masses),
        propagators(_propagators), vertices(_vertices) {}

private:
    std::vector<double> incoming_masses;
    std::vector<double> outgoing_masses;
    std::vector<Propagator> propagators;
    std::vector<Propagator> pseudo_propagators;
    std::vector<IndexList> vertices;
    std::vector<Propagator&> t_propagators;
    std::vector<std::vector<Decay>> s_decays;
    IndexList permutation;
    IndexList inverse_permutation;
}


class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(const Diagram& diagram, double nu=0., bool map_resonances=false)

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    std::vector<Invariant> t_invariants;
    std::vector<Invariant> s_pseudo_invariants;
};


class DiagramMapping : public Mapping {
public:
    DiagramMapping(const Diagram& diagram,
        const Diagram& diagram,
        double s_lab,
        double s_hat_min = 0.0,
        bool leptonic = false,
        double s_min_epsilon = 1e-2,
        double nu = 1.4
    ) : Mapping({scalar, scalar}, {scalar, scalar, scalar}, {}) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

};

}
