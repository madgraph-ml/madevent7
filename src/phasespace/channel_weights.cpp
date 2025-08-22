#include "madevent/phasespace/channel_weights.h"

#include "madevent/util.h"

using namespace madevent;

PropagatorChannelWeights::PropagatorChannelWeights(
    const std::vector<Topology>& topologies,
    const std::vector<std::vector<std::vector<std::size_t>>>& permutations,
    const std::vector<std::vector<std::size_t>> channel_indices
) :
    FunctionGenerator(
        "PropagatorChannelWeights",
        {batch_four_vec_array(topologies.at(0).outgoing_masses().size() + 2)},
        {batch_float_array([&](){
            std::size_t channel_count = 0;
            for (auto& perm : permutations) {
                channel_count += perm.size();
            }
            return channel_count;
        }())}
    )
{
    std::size_t channel_count = return_types().at(0).shape.at(0);
    _invariant_indices.resize(channel_count);
    _masses.resize(channel_count);
    _widths.resize(channel_count);

    std::map<std::vector<int64_t>, std::size_t> found_factors;
    std::size_t max_propagator_count = 0;
    for (auto [topology, chan_perms, indices] : zip(
        topologies, permutations, channel_indices
    )) {
        auto mom_terms = topology.propagator_momentum_terms();
        if (mom_terms.size() > max_propagator_count) {
            max_propagator_count = mom_terms.size();
        }
        for (auto [perm, index] : zip(chan_perms, indices)) {
            auto& chan_invariants = _invariant_indices.at(index);
            auto& chan_masses = _masses.at(index);
            auto& chan_widths = _widths.at(index);
            for (auto [factors, mass, width] : mom_terms) {
                std::vector<int64_t> permuted_factors;
                for (std::size_t i : perm) {
                    permuted_factors.push_back(factors.at(i));
                }
                auto found = found_factors.find(permuted_factors);
                std::size_t inv_index;
                if (found == found_factors.end()) {
                    inv_index = _momentum_factors.size();
                    _momentum_factors.emplace_back(
                        permuted_factors.begin(), permuted_factors.end()
                    );
                    found_factors[permuted_factors] = inv_index;
                } else {
                    inv_index = found->second;
                }
                chan_masses.push_back(mass);
                chan_widths.push_back(width);
                chan_invariants.push_back(inv_index);
            }
        }
    }
    for (auto& masses : _masses) masses.resize(max_propagator_count);
    for (auto& widths : _widths) widths.resize(max_propagator_count);
    for (auto& invars : _invariant_indices) invars.resize(max_propagator_count, -1);
}

ValueVec PropagatorChannelWeights::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto p_ext = args.at(0);
    auto invariants = fb.invariants_from_momenta(p_ext, _momentum_factors);
    auto channel_weights = fb.sde2_channel_weights(
        invariants, _masses, _widths, _invariant_indices
    );
    return {channel_weights};
}
