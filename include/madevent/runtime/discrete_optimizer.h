#pragma once

#include "madevent/madcode.h"
#include "madevent/runtime/context.h"
#include "madevent/runtime/tensor.h"

namespace madevent {

class DiscreteOptimizer {
public:
    DiscreteOptimizer(
        ContextPtr context, const std::vector<std::string>& prob_names
    ) :
        _context(context),
        _prob_names(prob_names),
        _sample_count(7000)
    {}
    void add_data(const std::vector<Tensor>& values_and_counts);
    void optimize();

private:
    ContextPtr _context;
    std::vector<std::string> _prob_names;
    double _damping;
    std::size_t _sample_count;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

}
