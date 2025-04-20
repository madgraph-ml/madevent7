#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/runtime/context.h"

namespace madevent {

class DiscreteSampler : public Mapping {
public:
    DiscreteSampler(
        std::vector<std::size_t> option_counts,
        const std::string& prefix = "",
        std::vector<std::size_t> dims_with_prior = {}
    );
    const std::vector<std::size_t>& option_counts() const { return _option_counts; }
    void initialize_globals(ContextPtr context) const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    std::vector<std::size_t> _option_counts;
    std::vector<std::size_t> _dims_with_prior;
    std::vector<std::string> _prob_names;
};

}
