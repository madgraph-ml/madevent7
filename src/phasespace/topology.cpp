#include "madevent/phasespace/topology.h"

#include <algorithm>

using namespace madevent;

namespace {

std::size_t next_vertex(const IndexList& vertices, std::size_t index) {
    return vertices[0] == index ? vertices[1] : vertices[0];
}

}

Diagram::LineRef::LineRef(std::string str) {
    if (str.size() < 2) {
        throw std::invalid_argument("Invalid line index");
    }
    switch (str.front()) {
    case 'i':
        type = Diagram::incoming;
        break;
    case 'o':
        type = Diagram::outgoing;
        break;
    case 'p':
        type = Diagram::propagator;
        break;
    default:
        throw std::invalid_argument("Invalid line type");
    }
    index = std::stoul(str.substr(1));
}

Diagram::Diagram(
    std::vector<double>& _incoming_masses,
    std::vector<double>& _outgoing_masses,
    std::vector<Propagator>& _propagators,
    std::vector<Vertex>& _vertices
) :
    incoming_masses(_incoming_masses),
    outgoing_masses(_outgoing_masses),
    propagators(_propagators),
    vertices(_vertices),
    incoming_vertices{-1, -1},
    outgoing_vertices(_outgoing_masses.size(), -1),
    propagator_vertices(_propagators.size()),
    decays(_propagators.size())
{
    if (incoming_masses.size() != 2) {
        throw std::invalid_argument("Diagram must have two incoming particles");
    }
    if (outgoing_masses.size() < 2) {
        throw std::invalid_argument("Diagram must have at least two outgoing particles");
    }

    std::size_t index = 0;
    for (auto& vertex : vertices) {
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

    find_s_and_t(incoming_vertices[1], -1);
}

bool Diagram::find_s_and_t(std::size_t current_index, int source_propagator) {
    bool is_t_vertex = false;
    std::vector<Diagram::LineRef> out_lines;
    for (auto& line_ref : vertices[current_index]) {
        switch(line_ref.type) {
        case Diagram::incoming:
            if (line_ref.index == 0) {
                t_vertices.push_back(current_index);
                is_t_vertex = true;
            }
            break;
        case Diagram::outgoing:
            out_lines.push_back(line_ref);
            break;
        case Diagram::propagator:
            if (line_ref.index != source_propagator) {
                if (find_s_and_t(
                    next_vertex(propagator_vertices[line_ref.index], current_index), line_ref.index
                )) {
                    t_propagators.push_back(line_ref.index);
                    t_vertices.push_back(current_index);
                    is_t_vertex = true;
                } else {
                    out_lines.push_back(line_ref);
                }
            }
            break;
        }
    }
    if (is_t_vertex) {
        lines_after_t.push_back(out_lines);
    } else {
        decays[source_propagator] = out_lines;
    }
    return is_t_vertex;
}

std::ostream& madevent::operator<<(std::ostream& out, const Diagram::LineRef& value) {
    switch (value.type) {
        case Diagram::incoming: out << "i"; break;
        case Diagram::outgoing: out << "o"; break;
        case Diagram::propagator: out << "p"; break;
    }
    out << value.index;
    return out;
}

Topology::Topology(const Diagram& diagram, Topology::DecayMode decay_mode) :
    incoming_masses(diagram.incoming_masses), outgoing_masses(diagram.outgoing_masses)
{

}
