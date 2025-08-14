#pragma once

#include "madevent/madcode.h"
#include "madevent/runtime/context.h"
#include "madevent/runtime/tensor.h"

namespace madevent {

class DiscreteOptimizer {
public:
    DiscreteOptimizer(
        ContextPtr context, const std::vector<std::string>& prob_names, double damping
    ) :
        _context(context),
        _prob_names(prob_names),
        _damping(std::max(std::min(damping, 1.), 0.)),
        _sample_count(7000)
    {}
    void add_data(Tensor weights, const std::vector<Tensor>& inputs);
    void optimize();

private:
    ContextPtr _context;
    std::vector<std::string> _prob_names;
    double _damping;
    std::size_t _sample_count;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

}
