#pragma once

#include "madevent/phasespace/mapping.h"
#include "madevent/backend/context.h"

#include <format>

namespace madevent {

class VegasMapping : public Mapping {
public:
    VegasMapping(std::size_t dimension, std::size_t bin_count, const std::string& prefix = "") :
        Mapping(
            {batch_float_array(dimension)},
            {batch_float_array(dimension)},
            {}
        ),
        _dimension(dimension),
        _bin_count(bin_count),
        _grid_name(std::format("{}vegas_grid", prefix))
    {}
    const std::string& grid_name() const { return _grid_name; }
    void initialize_global(ContextPtr context) const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override;

    std::size_t _dimension;
    std::size_t _bin_count;
    std::string _grid_name;
};

void initialize_vegas_grid(ContextPtr context, const std::string& grid_name);

}
