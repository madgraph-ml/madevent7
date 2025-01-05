#pragma once

#include <vector>
#include <array>
#include <string>
#include <ostream>

namespace madevent {

struct Propagator {
    double mass;
    double width;
    bool operator==(const Propagator& other) const {
        return mass == other.mass && width == other.width;
    }
};

using IndexList = std::vector<std::size_t>;

struct Diagram {
    enum LineType { incoming, outgoing, propagator };
    struct LineRef {
        LineType type;
        std::size_t index;
        LineRef(std::string str);
    };
    using Vertex = std::vector<LineRef>;

    std::vector<double> incoming_masses;
    std::vector<double> outgoing_masses;
    std::vector<Propagator> propagators;
    std::vector<Vertex> vertices;
    std::array<int, 2> incoming_vertices;
    std::vector<int> outgoing_vertices;
    std::vector<IndexList> propagator_vertices;
    IndexList t_propagators;
    IndexList t_vertices;
    std::vector<std::vector<LineRef>> lines_after_t;
    std::vector<std::vector<LineRef>> decays;

    Diagram(
        std::vector<double>& _incoming_masses,
        std::vector<double>& _outgoing_masses,
        std::vector<Propagator>& _propagators,
        std::vector<Vertex>& _vertices
    );

private:
    bool find_s_and_t(std::size_t current_index, int source_propagator);
};

std::ostream& operator<<(std::ostream& out, const Diagram::LineRef& value);

struct Topology {
    enum DecayMode { no_decays, massive_decays, all_decays };
    enum ComparisonResult { equal, permuted, different};
    struct Decay {
        Propagator propagator;
        std::size_t child_count = 1;
        bool operator==(const Decay& other) const {
            return propagator == other.propagator && child_count == other.child_count;
        }
    };

    std::vector<double> incoming_masses;
    std::vector<double> outgoing_masses;
    std::vector<Propagator> t_propagators;
    std::vector<std::vector<Decay>> decays;
    IndexList permutation;
    IndexList inverse_permutation;

    Topology(const Diagram& diagram, DecayMode decay_mode);
    ComparisonResult compare(const Topology& other, bool compare_t_propagators);

private:
    std::tuple<std::size_t, std::size_t> build_decays(
        const Diagram& diagram, DecayMode decay_mode, Diagram::LineRef line
    );
    void standardize_order(bool preserve_t_order);
};

}
