#pragma once
#include <vector>

namespace madevent {

struct Propagator {
    double mass;
    double width;
};

struct Diagram {
    enum LineType { incoming, outgoing, propagator };
    struct LineRef {
        LineType type;
        std::size_t index;
    };
    using Vertex = std::vector<LineRef>;

    std::vector<double> incoming_masses;
    std::vector<double> outgoing_masses;
    std::vector<Propagator> propagators;
    std::vector<Vertex> vertices;
};

struct Topology {
    enum DecayMode { no_decays, massive_decays, all_decays };
    struct Decay {
        Propagator propagator;
        std::size_t child_count;
    };
    using IndexList = std::vector<std::size_t>;

    std::vector<double> incoming_masses;
    std::vector<double> outgoing_masses;
    std::vector<Propagator> t_propagators;
    std::vector<std::vector<Decay>> s_decays;
    IndexList permutation;
    IndexList inverse_permutation;

    Topology(const Diagram& diagram, DecayMode decay_mode);
};

}
