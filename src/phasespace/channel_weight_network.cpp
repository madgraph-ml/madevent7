#include "madevent/phasespace/channel_weight_network.h"

using namespace madevent;

MomentumPreprocessing::MomentumPreprocessing(std::size_t particle_count) :
    FunctionGenerator(
        {batch_four_vec_array(particle_count), batch_float, batch_float},
        {batch_float_array(3 * (particle_count - 2) + 2)}
    ),
    _output_dim(3 * (particle_count - 2) + 2)
{}

ValueVec MomentumPreprocessing::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    return {fb.pt_eta_phi_x(args.at(0), args.at(1), args.at(2))};
}

ChannelWeightNetwork::ChannelWeightNetwork(
    std::size_t channel_count,
    std::size_t particle_count,
    std::size_t hidden_dim,
    std::size_t layers,
    MLP::Activation activation,
    const std::string& prefix
) :
    FunctionGenerator(
        {
            batch_four_vec_array(particle_count),
            batch_float,
            batch_float,
            batch_float_array(channel_count)
        },
        {batch_float_array(channel_count)}
    ),
    _preprocessing(particle_count),
    _mlp(
        _preprocessing.output_dim(),
        channel_count,
        hidden_dim,
        layers,
        activation,
        prefix
    )
{}

ValueVec ChannelWeightNetwork::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto p_ext = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);
    auto prior = args.at(3);
    auto net_input = _preprocessing.build_function(fb, {p_ext, x1, x2});
    auto net_output = _mlp.build_function(fb, net_input).at(0);
    return {fb.softmax_prior(net_output, prior)};
}
