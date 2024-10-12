#include "madevent/phasespace/topology.h"

#include <algorithm>

using namespace madevent;

namespace {

std::size_t next_vertex(const Topology::IndexList& vertices, std::size_t index) {
    return vertices[0] == index ? vertices[1] : vertices[0];
}

bool find_t_propagators(
    std::vector<Propagator>& t_propagators,
    const std::vector<Propagator>& propagators,
    const std::vector<int>& outgoing_vertices,
    const std::vector<Diagram::Vertex>& vertices,
    const std::vector<Topology::IndexList>& propagator_vertices,
    std::size_t current_index,
    int source_propagator
) {
    for (auto& line_ref : vertices[current_index]) {
        switch(line_ref.type) {
        case Diagram::incoming:
            if (line_ref.index == 0) return true;
            break;
        case Diagram::outgoing:
            return false;
        case Diagram::propagator:
            if (line_ref.index != source_propagator && find_t_propagators(
                t_propagators, propagators, outgoing_vertices, vertices, propagator_vertices,
                next_vertex(propagator_vertices[line_ref.index], current_index), line_ref.index
            )) {
                t_propagators.push_back(propagators[line_ref.index]);
                return true;
            }
            break;
        }
    }
    return false;
}

}

Topology::Topology(const Diagram& diagram, Topology::DecayMode decay_mode) :
    incoming_masses(diagram.incoming_masses), outgoing_masses(diagram.outgoing_masses)
{
    if (diagram.incoming_masses.size() != 2) {

    }
    if (diagram.outgoing_masses.size() < 2) {

    }

    std::vector<int> incoming_vertices(2, -1);
    std::vector<int> outgoing_vertices(diagram.outgoing_masses.size(), -1);
    std::vector<Topology::IndexList> propagator_vertices(diagram.propagators.size());
    std::size_t index;
    for (auto& vertex : diagram.vertices) {
        for (auto& line_ref : vertex) {
            switch (line_ref.type) {
            case Diagram::incoming:
                incoming_vertices[line_ref.index] = index;
                break;
            case Diagram::outgoing:
                outgoing_vertices[line_ref.index] = index;
                break;
            case Diagram::propagator:
                propagator_vertices[line_ref.index].push_back(index);
                break;
            }
        }
        ++index;
    }

    find_t_propagators(
        t_propagators, diagram.propagators, outgoing_vertices, diagram.vertices,
        propagator_vertices, incoming_vertices[1], -1
    );
}
