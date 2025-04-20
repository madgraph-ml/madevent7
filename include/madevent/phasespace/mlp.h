#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/runtime/context.h"

namespace madevent {

class MLP : public FunctionGenerator {
public:
    enum Activation { leaky_relu, linear };
    MLP(
        std::size_t input_dim,
        std::size_t output_dim,
        std::size_t hidden_dim = 32,
        std::size_t layers = 3,
        Activation activation = leaky_relu,
        const std::string& prefix = ""
    );

    std::size_t input_dim() const { return _input_dim; }
    std::size_t output_dim() const { return _output_dim; }
    void initialize_globals(ContextPtr context) const;
private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::size_t _input_dim;
    std::size_t _output_dim;
    std::size_t _hidden_dim;
    std::size_t _layers;
    Activation _activation;
    std::string _prefix;
};

}
