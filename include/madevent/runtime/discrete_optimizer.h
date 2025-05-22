#pragma once

#include "madevent/madcode.h"
#include "madevent/runtime/context.h"
#include "madevent/runtime/tensor.h"

namespace madevent {

class DiscreteOptimizer {
public:
    DiscreteOptimizer(
        ContextPtr context, const std::vector<std::string>& prob_names
    ) : _context(context), _prob_names(prob_names), _weight_sums(prob_names.size()) {}
    void optimize(Tensor weights, std::vector<Tensor>& inputs);

private:
    ContextPtr _context;
    std::vector<std::string> _prob_names;
    std::vector<std::vector<double>> _weight_sums;
    std::size_t _sample_count;
};

}
