#pragma once

#include "madevent/phasespace/base.h"
#include "madevent/runtime/context.h"

namespace madevent {

class VegasMapping : public Mapping {
public:
    VegasMapping(
        std::size_t dimension, std::size_t bin_count, const std::string& prefix = ""
    ) :
        Mapping(
            {batch_float_array(dimension)},
            {batch_float_array(dimension)},
            {}
        ),
        _dimension(dimension),
        _bin_count(bin_count),
        _grid_name(prefixed_name(prefix, "vegas_grid"))
    {}
    const std::string& grid_name() const { return _grid_name; }
    void initialize_globals(ContextPtr context) const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    std::size_t _dimension;
    std::size_t _bin_count;
    std::string _grid_name;
};

void initialize_vegas_grid(ContextPtr context, const std::string& grid_name);

}
