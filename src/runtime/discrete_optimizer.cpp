#include "madevent/runtime/discrete_optimizer.h"

using namespace madevent;

void DiscreteOptimizer::optimize(Tensor weights, std::vector<Tensor>& inputs) {
    //TODO: check shapes
    auto weights_cpu = weights.cpu();
    auto weights_view = weights_cpu.view<double, 1>();
    _sample_count += weights_view.size();
    double min_prob = 1. / _sample_count;
    for (auto [prob_name, weight_sum, input] : zip(_prob_names, _weight_sums, inputs)) {
        auto input_cpu = input.cpu();
        auto input_view = input_cpu.view<int64_t, 1>();
        auto prob_global = _context->global(prob_name);
        bool is_cpu = _context->device() == cpu_device();
        auto prob = is_cpu ? prob_global : Tensor(DataType::dt_float, prob_global.shape());
        auto prob_view = prob.view<double, 2>()[0];

        weight_sum.resize(prob_view.size());
        double norm = 0.;
        for (std::size_t i = 0; i < weights_view.size(); ++i) {
            auto w = weights_view[i];
            weight_sum.at(input_view[i]) += w;
            norm += w;
        }
        // probability is at least min_prob = 1/N to prevent setting it to zero
        // in early iterations with few samples
        double corrected_norm = 0.;
        for (std::size_t i = 0; i < prob_view.size(); ++i) {
            auto prob_val = std::min(min_prob, weight_sum.at(i) / norm);
            prob_view[i] = prob_val;
            corrected_norm += prob_val;
        }
        // normalize again to account for min_prob
        for (std::size_t i = 0; i < prob_view.size(); ++i) {
            prob_view[i] = prob_view[i] / corrected_norm;
        }

        if (!is_cpu) {
            prob_global.copy_from(prob);
        }
    }
}
