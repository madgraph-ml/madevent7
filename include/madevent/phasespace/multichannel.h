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
            _mappings.at(0).input_types(),
            _mappings.at(0).output_types(),
            [&] {
                auto condition_types = _mappings.at(0).condition_types();
                condition_types.push_back(multichannel_batch_size(_mappings.size()));
                return condition_types;
            }()
        ),
        mappings(_mappings)
    {}
private:
    Result build_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions, bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<PhaseSpaceMapping> mappings;
};

}
