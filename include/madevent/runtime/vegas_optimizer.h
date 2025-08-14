#pragma once

#include "madevent/madcode.h"
#include "madevent/runtime/context.h"
#include "madevent/runtime/tensor.h"

namespace madevent {

class VegasGridOptimizer {
public:
    VegasGridOptimizer(
        ContextPtr context, const std::string& grid_name, double damping
    ) : _context(context), _grid_name(grid_name), _damping(damping) {}
    void add_data(Tensor weights, Tensor inputs);
    void optimize();
    std::size_t input_dim() const;

private:
    ContextPtr _context;
    std::string _grid_name;
    double _damping;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

}
