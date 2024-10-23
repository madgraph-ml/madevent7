#pragma once

#include <vector>

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
        condition_types.push_back(scalar_int_array(_mappings.size()));
    }
private:
    Result build_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions, bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<PhaseSpaceMapping> mappings;
};

}
