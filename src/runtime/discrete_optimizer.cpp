#include "madevent/runtime/discrete_optimizer.h"

using namespace madevent;

void DiscreteOptimizer::optimize(Tensor weights, const std::vector<Tensor>& inputs) {
    //TODO: check shapes
    auto weights_cpu = weights.cpu();
    auto weights_view = weights_cpu.view<double, 1>();
    std::size_t next_sample_count = _sample_count + weights_view.size();
    double prob_ratio = 1.;//static_cast<double>(_sample_count) / next_sample_count;
    _sample_count = next_sample_count;
    for (auto [prob_name, input] : zip(_prob_names, inputs)) {
        auto input_cpu = input.cpu();
        auto input_view = input_cpu.view<int64_t, 1>();
        auto prob_global = _context->global(prob_name);
        bool is_cpu = _context->device() == cpu_device();
        auto prob = is_cpu ? prob_global : Tensor(DataType::dt_float, prob_global.shape());
        auto prob_view = prob.view<double, 2>()[0];
        auto option_count = prob_view.size();

        std::vector<double> weight_sums(option_count);
        std::vector<std::size_t> counts(prob_view.size());
        for (std::size_t i = 0; i < weights_view.size(); ++i) {
            auto w = weights_view[i];
            std::size_t index = input_view[i];
            weight_sums.at(index) += w;
            ++counts.at(index);
        }

        double norm = 0.;
        for (std::size_t i = 0; auto [wsum, count] : zip(weight_sums, counts)) {
            if (count > 0) wsum *= prob_view[i] / count;
            norm += wsum;
            ++i;
        }

        for (std::size_t i = 0; double wsum : weight_sums) {
            prob_view[i] = prob_view[i] * prob_ratio + wsum / norm * (1. - prob_ratio);
            ++i;
        }

        if (!is_cpu) {
            prob_global.copy_from(prob);
        }
    }
}
