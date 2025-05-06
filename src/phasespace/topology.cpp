#include "madevent/phasespace/topology.h"

#include <numeric>
#include <iostream>
#include <algorithm>

using namespace madevent;

namespace {

std::size_t next_vertex(const IndexVec& vertices, std::size_t index) {
    return vertices.at(0) == index ? vertices.at(1) : vertices.at(0);
}

std::size_t xorshift(std::size_t n, int i){
  return n^(n>>i);
}

template<class T>
std::size_t hash_combine(std::size_t seed, const T& value) {
    // combine hashes, based on https://stackoverflow.com/questions/35985960
    return std::rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ (
        17316035218449499591ull * xorshift(
            0x5555555555555555ull * xorshift(std::hash<T>{}(value), 32), 32
        )
    );
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
    const std::vector<double>& _incoming_masses,
    const std::vector<double>& _outgoing_masses,
    const std::vector<Propagator>& _propagators,
    const std::vector<Vertex>& _vertices
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
                incoming_vertices.at(line_ref.index) = index;
                break;
            case Diagram::outgoing:
                outgoing_vertices.at(line_ref.index) = index;
                break;
            case Diagram::propagator:
                propagator_vertices.at(line_ref.index).push_back(index);
                break;
            }
        }
        ++index;
    }

    find_s_and_t(incoming_vertices.at(1), -1);
}

bool Diagram::find_s_and_t(std::size_t current_index, int source_propagator) {
    bool is_t_vertex = false;
    std::vector<Diagram::LineRef> out_lines;
    for (auto& line_ref : vertices.at(current_index)) {
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
                    next_vertex(propagator_vertices.at(line_ref.index), current_index), line_ref.index
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
        decays.at(source_propagator) = out_lines;
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
    incoming_masses(diagram.incoming_masses),
    outgoing_masses(diagram.outgoing_masses),
    permutation(diagram.outgoing_masses.size())
{
    std::size_t max_depth = 0;
    std::size_t count_before = 0;
    auto prop_iter = diagram.t_propagators.begin();
    for (auto& lines : diagram.lines_after_t) {
        for (auto& line : lines) {
            auto [depth, count] = build_decays(diagram, decay_mode, line);
            for (; max_depth < depth; ++max_depth) {
                decays.at(max_depth).insert(decays.at(max_depth).begin(), count_before, Decay());
            }
            for (; depth < max_depth; ++depth) {
                decays.at(depth).insert(decays.at(depth).end(), count, Decay());
            }
            for (int i = 0; i < count - 1; ++i) {
                t_propagators.emplace_back();
            }
            count_before += count;
        }
        if (prop_iter != diagram.t_propagators.end()) {
            t_propagators.push_back(diagram.propagators.at(*prop_iter));
            ++prop_iter;
        }
    }
    standardize_order(decay_mode == DecayMode::all_decays);

    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(
        permutation.begin(),
        permutation.end(),
        [this](auto i, auto j) { return inverse_permutation.at(i) < inverse_permutation.at(j); }
    );

    std::size_t seed = 0;
    seed = hash_combine(seed, incoming_masses.size());
    for (double m : incoming_masses) seed = hash_combine(seed, m);
    seed = hash_combine(seed, outgoing_masses.size());
    for (double m : outgoing_masses) seed = hash_combine(seed, m);
    seed = hash_combine(seed, t_propagators.size());
    /*for (auto& prop : t_propagators) {
        seed = hash_combine(seed, prop.mass);
        seed = hash_combine(seed, prop.width);
    }*/
    seed = hash_combine(seed, decays.size());
    for (auto& layer : decays) {
        seed = hash_combine(seed, layer.size());
        for (auto& decay : layer) {
            seed = hash_combine(seed, decay.propagator.mass);
            seed = hash_combine(seed, decay.propagator.width);
            seed = hash_combine(seed, decay.child_count);
        }
    }
    decay_hash = seed;
}

Topology::ComparisonResult Topology::compare(const Topology& other, bool compare_t_propagators) const {
    if (
        (
            compare_t_propagators ?
            t_propagators != other.t_propagators :
            t_propagators.size() != other.t_propagators.size()
        ) ||
        decays != other.decays ||
        incoming_masses != other.incoming_masses ||
        outgoing_masses != other.outgoing_masses
    ) {
        return ComparisonResult::different;
    }

    if (permutation == other.permutation) {
        return ComparisonResult::equal;
    } else {
        return ComparisonResult::permuted;
    }
}

void Topology::standardize_order(bool preserve_t_order) {
    struct DecayTree {
        std::optional<Decay> decay;
        std::size_t min_index;
        std::vector<DecayTree*> nodes;

        void sort() {
            std::sort(
                nodes.begin(),
                nodes.end(),
                [this](auto tree1, auto tree2) {
                    if (!tree1->decay) {
                        return tree1->min_index < tree2->min_index;
                    }
                    return std::tie(
                        tree1->decay->child_count,
                        tree1->decay->propagator.mass,
                        tree1->decay->propagator.width,
                        tree1->min_index
                    ) < std::tie(
                        tree2->decay->child_count,
                        tree2->decay->propagator.mass,
                        tree2->decay->propagator.width,
                        tree2->min_index
                    );
                }
            );
        }

        void write_decays(
            std::vector<std::vector<Decay>>& decays,
            IndexVec& inverse_permutation,
            int depth
        ) {
            if (depth == -1) {
                inverse_permutation.push_back(min_index);
                return;
            }
            for (auto tree : nodes) {
                tree->write_decays(decays, inverse_permutation, depth - 1);
            }
            if (depth != decays.size()) {
                decays.at(depth).push_back(*decay);
            }
        }
    };

    std::vector<DecayTree> tree;
    std::size_t capacity = inverse_permutation.size() + 1;
    for (auto& layer : decays) {
        capacity += layer.size();
    }
    tree.reserve(capacity);
    for (auto index : inverse_permutation) {
        tree.push_back({std::nullopt, index, {}});
    }
    auto tree_iter = tree.begin();
    auto max_index = inverse_permutation.size();
    std::vector<DecayTree*> nodes;
    for (auto& layer : decays) {
        for (auto& decay : layer) {
            auto min_index = max_index;
            nodes.clear();
            for (std::size_t i = 0; i < decay.child_count; ++i, ++tree_iter) {
                if (tree_iter->min_index < min_index) {
                    min_index = tree_iter->min_index;
                }
                nodes.push_back(&*tree_iter);
            }
            tree.push_back({decay, min_index, nodes});
            tree.back().sort();
        }
        layer.clear();
    }
    nodes.clear();
    for (; tree_iter != tree.end(); ++tree_iter) {
        nodes.push_back(&*tree_iter);
    }
    tree.push_back({std::nullopt, 0, nodes});
    if (!preserve_t_order) {
        std::fill(t_propagators.begin(), t_propagators.end(), Propagator{});
        tree.back().sort();
    }
    inverse_permutation.clear();
    tree.back().write_decays(decays, inverse_permutation, decays.size());
}

std::tuple<std::size_t, std::size_t> Topology::build_decays(
    const Diagram& diagram, DecayMode decay_mode, Diagram::LineRef line_in
) {
    if (line_in.type == Diagram::outgoing) {
        inverse_permutation.push_back(line_in.index);
        return {0, 1};
    }

    auto& propagator = diagram.propagators.at(line_in.index);
    std::size_t max_depth = 0;
    std::size_t count_before = 0;
    for (auto& line : diagram.decays.at(line_in.index)) {
        auto [depth, count] = build_decays(diagram, decay_mode, line);
        for (; max_depth < depth; ++max_depth) {
            decays.at(max_depth).insert(decays.at(max_depth).begin(), count_before, Decay());
        }
        for (; depth < max_depth; ++depth) {
            decays.at(depth).insert(decays.at(depth).end(), count, Decay());
        }
        count_before += count;
    }

    // TODO maybe not check for mass 0 but for propagator mass > minimum mass
    if (decay_mode == Topology::all_decays ||
        (decay_mode == Topology::massive_decays && propagator.mass != 0)
    ) {
        if (decays.size() == max_depth) {
            decays.emplace_back();
        }
        decays.at(max_depth).push_back({propagator, count_before});
        ++max_depth;
        count_before = 1;
    }
    return {max_depth, count_before};
}
