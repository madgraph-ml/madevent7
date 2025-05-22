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
    void optimize(Tensor weights, Tensor inputs);
    std::size_t input_dim() const;

private:
    ContextPtr _context;
    std::string _grid_name;
    double _damping;
};

}
