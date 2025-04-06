#include "madevent/phasespace/flow.h"

#include <numeric>
#include <algorithm>
#include <bitset>
#include <format>

using namespace madevent;

namespace {

std::tuple<Value, Value> build_block(
    FunctionBuilder& fb,
    const MLP& subnet,
    Value input,
    Value condition,
    int64_t bin_count,
    bool inverse
) {
    auto subnet_out = subnet.build_function(fb, {condition});
    auto [widths, heights, derivatives] = fb.rqs_activation(subnet_out.at(0), bin_count);
    auto rqs_condition = fb.rqs_find_bin(
        input, inverse ? heights : widths, inverse ? widths : heights, derivatives
    );
    auto [out, det] = inverse ?
        fb.rqs_inverse(input, rqs_condition) :
        fb.rqs_forward(input, rqs_condition);
    return {out, fb.product(det)};
}

}

Flow::Flow(
    std::size_t input_dim,
    std::size_t condition_dim,
    const std::string& prefix,
    std::size_t bin_count,
    std::size_t subnet_hidden_dim,
    std::size_t subnet_layers,
    MLP::Activation subnet_activation,
    bool invert_spline
) :
    Mapping(
        {batch_float_array(input_dim)},
        {batch_float_array(input_dim)},
        condition_dim == 0 ? TypeVec{} : TypeVec{batch_float_array(condition_dim)}
    ),
    _input_dim(input_dim),
    _condition_dim(condition_dim),
    _bin_count(bin_count),
    _invert_spline(invert_spline)
{
    std::size_t block_count = 0;
    for(std::size_t dim = input_dim; dim > 0; dim /= 2, ++block_count);
    std::vector<std::bitset<32>> masks;
    for (std::size_t i = 0; i < input_dim; ++i) masks.push_back(i);

    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        std::vector<int64_t> indices1, indices2;
        std::size_t dim_index = 0;
        for (auto& mask : masks) {
            if (mask.test(block_index)) {
                indices1.push_back(dim_index);
            } else {
                indices2.push_back(dim_index);
            }
            ++dim_index;
        }
        _coupling_blocks.emplace_back(
            MLP(
                indices2.size() + condition_dim,
                indices1.size() * (3 * bin_count + 1),
                subnet_hidden_dim,
                subnet_layers,
                subnet_activation,
                std::format("{}subnet{}a_", prefix, block_index + 1)
            ),
            MLP(
                indices1.size() + condition_dim,
                indices2.size() * (3 * bin_count + 1),
                subnet_hidden_dim,
                subnet_layers,
                subnet_activation,
                std::format("{}subnet{}b_", prefix, block_index + 1)
            ),
            indices1,
            indices2
        );
    }
}

void Flow::initialize_globals(ContextPtr context) const {
    for (auto& block : _coupling_blocks) {
        block.subnet1.initialize_globals(context);
        block.subnet2.initialize_globals(context);
    }
}

Mapping::Result Flow::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    return build_transform(fb, inputs, conditions, false);
}

Mapping::Result Flow::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    return build_transform(fb, inputs, conditions, true);
}

Mapping::Result Flow::build_transform(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions, bool inverse
) const {
    Value x = inputs.at(0);
    Value cond;
    bool has_cond = _condition_dim != 0;
    if (has_cond) cond = conditions.at(0);
    ValueVec dets;
    std::vector<int64_t> dim_positions(_input_dim);
    std::iota(dim_positions.begin(), dim_positions.end(), 0);

    auto loop_body = [&](const CouplingBlock& block) {
        auto half1 = fb.select(x, block.indices1);
        auto half2 = fb.select(x, block.indices2);
        Value det1, det2;
        bool spline_inv = _invert_spline ^ inverse;
        if (inverse) {
            auto cond1 = has_cond ? fb.cat({half2, cond}) : half2;
            std::tie(half1, det1) = build_block(
                fb, block.subnet1, half1, cond1, _bin_count, spline_inv
            );
            auto cond2 = has_cond ? fb.cat({half1, cond}) : half1;
            std::tie(half2, det2) = build_block(
                fb, block.subnet2, half2, cond2, _bin_count, spline_inv
            );
        } else {
            auto cond2 = has_cond ? fb.cat({half1, cond}) : half1;
            std::tie(half2, det2) = build_block(
                fb, block.subnet2, half2, cond2, _bin_count, spline_inv
            );
            auto cond1 = has_cond ? fb.cat({half2, cond}) : half2;
            std::tie(half1, det1) = build_block(
                fb, block.subnet1, half1, cond1, _bin_count, spline_inv
            );
        }
        dets.push_back(det1);
        dets.push_back(det2);
        x = fb.cat({half1, half2});
        auto iter_next = std::copy(
            block.indices1.begin(), block.indices1.end(), dim_positions.begin()
        );
        std::copy(block.indices2.begin(), block.indices2.end(), iter_next);
    };
    if (inverse) {
        std::for_each(_coupling_blocks.rbegin(), _coupling_blocks.rend(), loop_body);
    } else {
        std::for_each(_coupling_blocks.begin(), _coupling_blocks.end(), loop_body);
    }

    return {
        {fb.select(x, dim_positions)},
        fb.product(fb.stack(dets))
    };
}
