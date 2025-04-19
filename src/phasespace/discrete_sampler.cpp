#include "madevent/phasespace/discrete_sampler.h"

using namespace madevent;

DiscreteSampler::DiscreteSampler(
    std::vector<std::size_t> option_counts,
    const std::string& prefix,
    std::vector<std::size_t> dims_with_prior
) :
    Mapping(
        {batch_float_array(option_counts.size())},
        TypeVec(option_counts.size(), batch_float),
        {}
    ),
    _option_counts(option_counts),
    _dims_with_prior(dims_with_prior)
{
    for (std::size_t i = 0; i < option_counts.size(); ++i) {
        _prob_names.push_back(std::format("{}prob{}", prefix, i));
    }
}

Mapping::Result DiscreteSampler::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {

}

Mapping::Result DiscreteSampler::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {

}

void DiscreteSampler::initialize_globals(ContextPtr context) const {

}
