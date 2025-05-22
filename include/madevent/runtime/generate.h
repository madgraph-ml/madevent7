#pragma once

#include <optional>
#include <vector>

#include "madevent/madcode.h"
#include "madevent/phasespace.h"
#include "madevent/runtime/io.h"
#include "madevent/runtime/vegas_optimizer.h"
#include "madevent/runtime/runtime_base.h"

namespace madevent {

class RunningIntegral {
public:
    RunningIntegral() : _mean(0), _var_sum(0), _count(0) {}
    double mean() const {
        return _mean;
    }
    double variance() const {
        return _count > 1 ? _var_sum / (_count - 1) : 0;
    }
    double error() const {
        return std::sqrt(variance() / _count);
    }
    double rel_error() const {
        return error() / mean();
    }
    double rel_std_dev() const {
        return std::sqrt(variance()) / _mean;
    }
    std::size_t count() const { return _count; }
    void reset() { _mean = 0; _var_sum = 0; _count = 0; }
    void push(double value) {
        ++_count;
        if (_count == 1) {
            _mean = value;
            _var_sum = 0;
        } else {
            double mean_diff = value - _mean;
            _mean += mean_diff / _count;
            _var_sum += mean_diff * (value - _mean);
        }
    }
private:
    double _mean;
    double _var_sum;
    std::size_t _count;
};

class EventGenerator {
public:
    static inline const int integrand_flags =
        Integrand::sample | Integrand::return_momenta | Integrand::return_random;
    struct Config {
        std::size_t target_count = 10000;
        double vegas_damping = 0.2;
        double max_overweight_fraction = 0.01;
        double max_overweight_truncation = 0.01;
        std::size_t start_batch_size = 1000;
        std::size_t max_batch_size = 64000;
        std::size_t survey_min_iters = 3;
        std::size_t survey_max_iters = 4;
        double survey_target_precision = 0.1;
        std::size_t optimization_patience = 3;
        double optimization_threshold = 0.99;
    };
    static const Config default_config;
    struct Status {
        std::size_t index;
        double mean;
        double error;
        double rel_std_dev;
        std::size_t count;
        double count_unweighted;
        double count_target;
        std::size_t iterations;
        bool done;
    };

    EventGenerator(
        ContextPtr context,
        const std::vector<Integrand>& channels,
        const std::string& file_name,
        const Config& config = default_config,
        const std::optional<std::string>& temp_file_dir = std::nullopt
    );
    void survey();
    void generate();
    Status status() const { return _status_all; }
    std::vector<Status> channel_status() const;
private:
    struct ChannelState {
        std::size_t index;
        RuntimePtr runtime;
        EventFile writer;
        std::optional<VegasGridOptimizer> vegas_optimizer;
        std::size_t batch_size;
        RunningIntegral cross_section;
        bool needs_optimization = true;
        double max_weight = 0.;
        double eff_count = 0.;
        double integral_fraction = 1.;
        std::size_t total_sample_count = 0;
        std::size_t iterations = 0;
        std::size_t iters_without_improvement = 0;
        double best_rsd = std::numeric_limits<double>::max();
    };

    ContextPtr _context;
    Config _config;
    std::vector<ChannelState> _channels;
    std::vector<std::tuple<double, double, std::size_t>> _large_weights;
    double _max_weight;
    RuntimePtr _unweighter;
    Status _status_all;
    EventFile _writer;

    void unweight_all();
    void combine();
    std::tuple<Tensor, std::vector<Tensor>> generate_channel(
        ChannelState& channel, bool always_optimize
    );
    void clear_channel(ChannelState& channel);
    void update_max_weight(ChannelState& channel, Tensor weights);
    void unweight_and_write(ChannelState& channel, const std::vector<Tensor>& momenta);
    void print_gen_init();
    void print_gen_update();
};

}
