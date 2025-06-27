#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/phasespace/mlp.h"

namespace madevent {

class Flow : public Mapping {
public:
    Flow(
        std::size_t input_dim,
        std::size_t condition_dim = 0,
        const std::string& prefix = "",
        std::size_t bin_count = 10,
        std::size_t subnet_hidden_dim = 32,
        std::size_t subnet_layers = 3,
        MLP::Activation subnet_activation = MLP::leaky_relu,
        bool invert_spline = true
    );
    std::size_t input_dim() const { return _input_dim; }
    std::size_t condition_dim() const { return _condition_dim; }
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

    struct CouplingBlock {
        MLP subnet1;
        MLP subnet2;
        std::vector<int64_t> indices1;
        std::vector<int64_t> indices2;
    };

    std::vector<CouplingBlock> _coupling_blocks;
    std::size_t _input_dim;
    std::size_t _condition_dim;
    std::size_t _bin_count;
    bool _invert_spline;
};

}
