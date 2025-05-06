#include "madevent/phasespace/discrete_flow.h"

#include "madevent/util.h"

using namespace madevent;

DiscreteFlow::DiscreteFlow(
    std::vector<std::size_t> option_counts,
    const std::string& prefix,
    const std::vector<std::size_t>& dims_with_prior,
    std::size_t condition_dim,
    std::size_t subnet_hidden_dim,
    std::size_t subnet_layers,
    MLP::Activation subnet_activation
) :
    Mapping(
        {batch_float_array(option_counts.size())},
        TypeVec(option_counts.size(), batch_float),
        condition_dim == 0 ? TypeVec{} : TypeVec{batch_float_array(condition_dim)}
    ),
    _option_counts(option_counts),
    _dims_with_prior(dims_with_prior),
    _condition_dim(condition_dim)
{
    std::size_t option_sum = 0, dim_index = 0;
    for (std::size_t option_count : option_counts) {
        std::size_t subnet_input_dim = option_sum + condition_dim;
        if (subnet_input_dim > 0) {
            _subnets.emplace_back(
                subnet_input_dim,
                option_count,
                subnet_hidden_dim,
                subnet_layers,
                subnet_activation,
                std::format("{}subnet{}_", prefix, dim_index + 1)
            );
        } else {
            _first_prob_name = std::format("{}prob0", prefix);
        }
        option_sum += option_count;
        ++dim_index;
    }
}

Mapping::Result DiscreteFlow::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto random_inputs = fb.unstack(inputs.at(0));

    ValueVec sampled_indices, dets;
    Value subnet_input;
    if (_condition_dim != 0) {
        subnet_input = conditions.at(0);
    }
    std::size_t dim_index = 0;
    int64_t prev_option_count = 0;
    auto mlp_iter = _subnets.begin();
    for (auto [option_count, random] : zip(_option_counts, random_inputs)) {
        Value probs;
        if (dim_index > 0) {
            subnet_input = fb.cat({
                subnet_input, fb.one_hot(sampled_indices.back(), prev_option_count)
            });
        }
        if (dim_index == 0 && _first_prob_name) {
            probs = fb.global(
                *_first_prob_name,
                DataType::dt_float,
                {static_cast<int>(option_count)}
            );
        } else {
            probs = mlp_iter->build_function(fb, {subnet_input}).at(0);
            mlp_iter++;
        }
        auto [index, det] = fb.sample_discrete_probs(random, probs);
        sampled_indices.push_back(index);
        dets.push_back(det);
        prev_option_count = option_count;
        ++dim_index;
    }
    Value det = dim_index == 1 ? dets.at(0) : fb.product(fb.stack(dets));
    return {sampled_indices, det};
}

Mapping::Result DiscreteFlow::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {

}

void DiscreteFlow::initialize_globals(ContextPtr context) const {

}
