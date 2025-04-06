#include "madevent/phasespace/vegas.h"

using namespace madevent;

Mapping::Result VegasMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto grid = fb.global(
        _grid_name,
        DataType::dt_float,
        {static_cast<int>(_dimension), static_cast<int>(_bin_count)}
    );
    auto [output, dets] = fb.vegas_forward(inputs.at(0), grid);
    return {{output}, fb.product(dets)};
}

Mapping::Result VegasMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto grid = fb.global(
        _grid_name,
        DataType::dt_float,
        {static_cast<int>(_dimension), static_cast<int>(_bin_count)}
    );
    auto [output, dets] = fb.vegas_inverse(inputs.at(0), grid);
    return {{output}, fb.product(dets)};
}

void VegasMapping::initialize_global(ContextPtr context) const {
    context->define_global(_grid_name, DataType::dt_float, {_dimension, _bin_count});
    initialize_vegas_grid(context, _grid_name);
}

void madevent::initialize_vegas_grid(ContextPtr context, const std::string& grid_name) {
    auto grid = context->global(grid_name);
    if (
        grid.shape().size() != 3 ||
        grid.size(0) != 1 ||
        grid.size(2) < 2 ||
        grid.dtype() != DataType::dt_float
    ) {
        throw std::runtime_error("Invalid vegas grid type");
    }
    auto grid_view = grid.view<double, 3>()[0];
    auto bin_count = grid.size(2) - 1;
    double increment = 1. / bin_count;
    for (std::size_t i = 0; i < grid_view.size(); ++i) {
        for (std::size_t j = 0; j < bin_count; ++j) {
            grid_view[i][j] = increment * j;
        }
        grid_view[i][bin_count] = 1.;
    }
}

