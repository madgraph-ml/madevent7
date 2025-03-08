#pragma once

#include <vector>
#include <format>

#include "madevent/phasespace/mapping.h"
#include "madevent/phasespace/phasespace.h"

namespace madevent {

class MultiChannelMapping : public Mapping {
public:
    MultiChannelMapping(const std::vector<PhaseSpaceMapping>& _mappings) :
        Mapping(
            _mappings.at(0).get_input_types(),
            _mappings.at(0).get_output_types(),
            _mappings.at(0).get_condition_types()
        ),
        mappings(_mappings)
    {
        std::vector<BatchSize> batch_sizes;
        BatchSize remaining = batch_size;
        for (std::size_t i = 0; i < _mappings.size() - 1; ++i) {
            BatchSize batch_size_i(std::format("channel_size_{}", i));
            batch_sizes.push_back(batch_size_i);
            remaining = remaining - batch_size_i;
        }
        batch_sizes.push_back(remaining);
        condition_types.push_back(batch_sizes);
    }
private:
    Result build_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions, bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<PhaseSpaceMapping> mappings;
};

}
