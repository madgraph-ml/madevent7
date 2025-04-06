#include "madevent/phasespace/multichannel.h"

using namespace madevent;


Mapping::Result MultiChannelMapping::build_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions, bool inverse
) const {
    auto& counts = conditions.back();

    std::vector<ValueVec> split_inputs;
    for (auto& input : inputs) {
        split_inputs.push_back(fb.batch_split(input, counts));
    }
    std::vector<ValueVec> split_conditions;
    for (auto& condition : conditions) {
        if (&condition == &counts) break;
        split_conditions.push_back(fb.batch_split(condition, counts));
    }

    std::vector<ValueVec> split_outputs(output_types().size());
    ValueVec split_dets;
    std::size_t index = 0;
    for (auto& mapping : mappings) {
        ValueVec in, cond;
        for (auto& input : split_inputs) {
            in.push_back(input[index]);
        }
        for (auto& condition : split_conditions) {
            cond.push_back(condition[index]);
        }
        auto [output, det] =
            inverse ?
            mapping.build_inverse(fb, in, cond) :
            mapping.build_forward(fb, in, cond);
        auto split_out_iter = split_outputs.begin();
        for (auto& out : output) {
            split_out_iter->push_back(out);
            ++split_out_iter;
        }
        split_dets.push_back(det);
        ++index;
    }
    ValueVec cat_outputs;
    for (auto& output : split_outputs) {
        auto [cat, _] = fb.batch_cat(output);
        cat_outputs.push_back(cat);
    }
    auto [det, _] = fb.batch_cat(split_dets);
    return {cat_outputs, det};
}
