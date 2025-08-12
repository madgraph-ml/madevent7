#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/runtime/context.h"

namespace madevent {

class DiscreteSampler : public Mapping {
public:
    DiscreteSampler(
        const std::vector<std::size_t>& option_counts,
        const std::string& prefix = "",
        const std::vector<std::size_t>& dims_with_prior = {}
    );
    const std::vector<std::size_t>& option_counts() const { return _option_counts; }
    const std::vector<std::string>& prob_names() const { return _prob_names; }
    void initialize_globals(ContextPtr context) const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_transform(
        FunctionBuilder& fb,
        const ValueVec& inputs,
        const ValueVec& conditions,
        bool inverse
    ) const;

    std::vector<std::size_t> _option_counts;
    std::vector<bool> _dim_has_prior;
    std::vector<std::string> _prob_names;
};

void initialize_uniform_probs(
    ContextPtr context, const std::string& name, std::size_t option_count
);

}
