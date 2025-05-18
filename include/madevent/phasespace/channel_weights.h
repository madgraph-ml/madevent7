#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/phasespace/topology.h"

namespace madevent {

class PropagatorChannelWeights : public FunctionGenerator {
public:
    PropagatorChannelWeights(
        const std::vector<Topology>& topologies,
        const std::vector<std::vector<std::vector<std::size_t>>>& permutations,
        const std::vector<std::vector<std::size_t>> channel_indices
    );

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<std::vector<double>> _momentum_factors;
    std::vector<std::vector<int64_t>> _invariant_indices;
    std::vector<std::vector<double>> _masses;
    std::vector<std::vector<double>> _widths;
};

}
