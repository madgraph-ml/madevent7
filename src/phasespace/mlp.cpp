#include "madevent/phasespace/mlp.h"

#include <format>
#include <random>

using namespace madevent;

namespace {

Value build_layer(
    FunctionBuilder& fb,
    Value input,
    int input_dim,
    int output_dim,
    MLP::Activation activation,
    const std::string& prefix,
    int layer_index
) {
    auto weight = fb.global(
        std::format("{}layer{}_weight", prefix, layer_index),
        DataType::dt_float,
        {output_dim, input_dim}
    );
    auto bias = fb.global(
        std::format("{}layer{}_bias", prefix, layer_index),
        DataType::dt_float,
        {output_dim}
    );
    auto linear_out = fb.matmul(input, weight, bias);
    switch (activation) {
    case MLP::leaky_relu:
        return fb.leaky_relu(linear_out);
    case MLP::linear:
        return linear_out;
    }
}

void initialize_layer(
    ContextPtr context,
    std::size_t input_dim,
    std::size_t output_dim,
    const std::string& prefix,
    int layer_index,
    std::mt19937& rand_gen,
    bool zeros
) {
    double bound = 1 / std::sqrt(input_dim);
    std::uniform_real_distribution<double> rand_dist(-bound, bound);
    auto weight_name = std::format("{}layer{}_weight", prefix, layer_index);
    auto bias_name = std::format("{}layer{}_bias", prefix, layer_index);
    auto weight_tensor = context->define_global(
        weight_name, DataType::dt_float, {output_dim, input_dim}, true
    );
    auto bias_tensor = context->define_global(
        bias_name, DataType::dt_float, {output_dim}, true
    );

    auto weight_view = weight_tensor.view<double, 3>()[0];
    for (std::size_t i = 0; i < output_dim; ++i) {
        for (std::size_t j = 0; j < input_dim; ++j) {
            weight_view[i][j] = zeros ? 0. : rand_dist(rand_gen);
        }
    }
    auto bias_view = bias_tensor.view<double, 2>()[0];
    for (std::size_t i = 0; i < output_dim; ++i) {
        bias_view[i] = zeros ? 0. : rand_dist(rand_gen);
    }
}

}

MLP::MLP(
    std::size_t input_dim,
    std::size_t output_dim,
    std::size_t hidden_dim,
    std::size_t layers,
    Activation activation,
    const std::string& prefix
) :
    FunctionGenerator({batch_float_array(input_dim)}, {batch_float_array(output_dim)}),
    _input_dim(input_dim),
    _output_dim(output_dim),
    _hidden_dim(hidden_dim),
    _layers(layers),
    _activation(activation),
    _prefix(prefix)
{};

ValueVec MLP::build_function_impl(FunctionBuilder& fb, const ValueVec& args) const {
    std::size_t dim = _input_dim;
    Value x = args.at(0);
    for (std::size_t i = 1; i < _layers; ++i) {
        x = build_layer(fb, x, dim, _hidden_dim, _activation, _prefix, i);
        dim = _hidden_dim;
    }
    return {build_layer(fb, x, dim, _output_dim, MLP::linear, _prefix, _layers)};
}

void MLP::initialize_globals(ContextPtr context) const {
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    std::size_t dim = _input_dim;
    for (std::size_t i = 1; i < _layers; ++i) {
        initialize_layer(context, dim, _hidden_dim, _prefix, i, rand_gen, false);
        dim = _hidden_dim;
    }
    initialize_layer(context, dim, _output_dim, _prefix, _layers, rand_gen, true);
}
