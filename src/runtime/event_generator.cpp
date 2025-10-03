#include "madevent/runtime/event_generator.h"

#include <filesystem>
#include <format>
#include <cmath>
#include <random>
#include <ranges>

#include "madevent/runtime/format.h"
#include "madevent/util.h"

using namespace madevent;
namespace fs = std::filesystem;

const EventGenerator::Config EventGenerator::default_config = {};

EventGenerator::EventGenerator(
    ContextPtr context,
    const std::vector<Integrand>& channels,
    const std::string& file_name,
    const Config& config,
    const std::optional<std::string>& temp_file_dir
) :
    _context(context),
    _config(config),
    _max_weight(0.),
    _unweighter(build_runtime(
        Unweighter(
            {channels.at(0).return_types().at(0), channels.at(0).return_types().at(1)}
        ).function(),
        context,
        false
    )),
    _status_all(
        {0, 0., 0., 0., 0, 0, 0., static_cast<double>(config.target_count), false, false}
    ),
    _writer(
        file_name,
        std::ranges::max(std::views::transform(channels, [] (auto& chan) {
            return chan.particle_count();
        }))
    ),
    _job_id(0)
{
    std::size_t i = 0;
    fs::path file_path(file_name);
    fs::path temp_path = temp_file_dir.value_or(file_path.parent_path());
    for (auto& channel : channels) {
        if (channel.flags() != integrand_flags) {
            throw std::invalid_argument(
                "Integrand flags must be sample | return_momenta | return_random | return_discrete"
            );
        }
        auto chan_path = temp_path / file_path.stem();
        chan_path += std::format(".channel{}.npy", i);
        std::optional<VegasGridOptimizer> vegas_optimizer;
        RuntimePtr vegas_histogram = nullptr;
        if (const auto& name = channel.vegas_grid_name(); name) {
            vegas_optimizer = VegasGridOptimizer(context, *name, config.vegas_damping);
            VegasHistogram hist(channel.vegas_dimension(), channel.vegas_bin_count());
            vegas_histogram = build_runtime(hist.function(), context, false);
        }
        std::vector<std::string> prob_names;
        std::vector<std::size_t> option_counts;
        auto add_names = Overloaded {
            [&](const DiscreteSampler& sampler) {
                auto& names = sampler.prob_names();
                prob_names.insert(prob_names.end(), names.begin(), names.end());
                auto& opts = sampler.option_counts();
                option_counts.insert(option_counts.end(), opts.begin(), opts.end());
            },
            [](auto sampler) {}
        };
        std::visit(add_names, channel.discrete_before());
        std::visit(add_names, channel.discrete_after());
        std::optional<DiscreteOptimizer> discrete_optimizer;
        RuntimePtr discrete_histogram = nullptr;
        if (prob_names.size() > 0) {
            discrete_optimizer = DiscreteOptimizer(context, prob_names);
            DiscreteHistogram hist(option_counts);
            discrete_histogram = build_runtime(hist.function(), context, false);
        }
        _channels.push_back({
            i,
            build_runtime(channel.function(), context, false),
            EventFile(chan_path.string(), channel.particle_count(), EventFile::create, true),
            vegas_optimizer,
            std::move(vegas_histogram),
            discrete_optimizer,
            std::move(discrete_histogram),
            config.start_batch_size
        });
        ++i;
    }
}

void EventGenerator::survey() {
    bool done = false;
    auto& thread_pool = default_thread_pool();
    std::size_t min_iters = _config.survey_min_iters;
    std::size_t max_iters = std::max(min_iters, _config.survey_max_iters);
    double target_precision = _config.survey_target_precision;

    for (std::size_t iter = 0; !done && iter < max_iters; ++iter) {
        for (auto& channel : _channels) {
            if (iter >= min_iters && channel.cross_section.rel_error() < target_precision) {
                continue;
            }
            clear_channel(channel);
            start_vegas_jobs(channel);
        }
        done = true;
        while (auto job_id = thread_pool.wait()) {
            _abort_check_function();
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            --channel.job_count;

            bool run_optim = channel.vegas_optimizer || channel.discrete_optimizer;
            auto [weights, events] = integrate_and_optimize(channel, job.events, run_optim);

            if (iter >= min_iters - 1) {
                update_max_weight(channel, weights);
                unweight_and_write(channel, events);
                if (
                    channel.job_count == 0 &&
                    channel.cross_section.rel_error() < target_precision
                ) {
                    done = false;
                }
            } else {
                done = false;
            }
            _running_jobs.erase(*job_id);
        }
    }
    println("survey: {} +- {}", _status_all.mean, _status_all.error);
}

void EventGenerator::generate() {
    print_gen_init();
    auto& thread_pool = default_thread_pool();
    std::size_t channel_index = 0;
    std::size_t target_job_count = 2 * thread_pool.thread_count();
    while (true) {
        _abort_check_function();

        std::size_t job_count_before;
        do {
            job_count_before = _running_jobs.size();
            for (
                std::size_t i = 0;
                i < _channels.size() && _running_jobs.size() < target_job_count;
                ++i, channel_index = (channel_index + 1) % _channels.size()
            ) {
                auto& channel = _channels.at(channel_index);
                if (channel.eff_count >= channel.integral_fraction * _config.target_count) {
                    continue;
                }
                if (
                    (channel.vegas_optimizer || channel.discrete_optimizer) &&
                    channel.needs_optimization
                ) {
                    if (channel.job_count == 0) start_vegas_jobs(channel);
                } else {
                    start_job(channel, _config.batch_size);
                }
            }
        } while (_running_jobs.size() - job_count_before > 0);

        if (auto job_id = thread_pool.wait()) {
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            bool run_optim =
                (channel.vegas_optimizer || channel.discrete_optimizer) &&
                channel.needs_optimization;
            if (run_optim && channel.job_count == job.vegas_job_count) {
                clear_channel(channel);
            }
            --channel.job_count;

            auto [weights, events] = integrate_and_optimize(channel, job.events, run_optim);
            update_max_weight(channel, weights);
            unweight_and_write(channel, events);
            print_gen_update();
            _running_jobs.erase(*job_id);
        } else {
            if (_status_all.done) unweight_all();
            if (_status_all.done) break;
        }
    }
    combine();
}

void EventGenerator::unweight_all() {
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    std::uniform_real_distribution<double> rand_dist;
    bool done = true;
    double total_eff_count = 0.;
    for (auto& channel : _channels) {
        channel.max_weight = std::max(
            channel.max_weight, _max_weight * channel.integral_fraction
        );
        auto ecb = channel.eff_count;
        channel.eff_count = channel.writer.unweight(
            channel.max_weight, [&]() { return rand_dist(rand_gen); }
        );

        double chan_target = channel.integral_fraction * _config.target_count;
        if (channel.eff_count < chan_target) {
            total_eff_count += channel.eff_count;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status_all.count_unweighted = total_eff_count;
    _status_all.done = done;
}

void EventGenerator::combine() {
    std::vector<std::size_t> channel_counts;
    std::size_t count_sum = 0;
    for (auto& channel : _channels) {
        std::size_t count = std::round(channel.integral_fraction * _config.target_count);
        count_sum += count;
        channel_counts.push_back(count_sum);
        channel.writer.seek(0);
    }

    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    EventBuffer buffer(_writer.particle_count());
    while (channel_counts.back() > 0) {
        auto index = std::uniform_int_distribution<std::size_t>(
            0, channel_counts.back() - 1
        )(rand_gen);
        auto channel_iter = std::lower_bound(
            channel_counts.begin(), channel_counts.end(), index
        );
        std::for_each(channel_iter, channel_counts.end(), [](auto& count) { --count; });
        auto& channel = _channels.at(channel_iter - channel_counts.begin());
        auto& writer = channel.writer;
        do {
            writer.read(buffer);
        } while(buffer.event().weight == 0);
        buffer.event().weight = std::max(
            1., buffer.event().weight / channel.max_weight
        );
        _writer.write(buffer);
    }
}

std::vector<EventGenerator::Status> EventGenerator::channel_status() const {
    std::vector<EventGenerator::Status> status;
    for (auto& channel : _channels) {
        double target_count = channel.integral_fraction * _config.target_count;
        status.push_back({
            channel.index,
            channel.cross_section.mean(),
            channel.cross_section.error(),
            channel.cross_section.rel_std_dev(),
            channel.total_sample_count,
            channel.cross_section.count(),
            channel.eff_count,
            target_count,
            channel.iterations,
            !channel.needs_optimization,
            channel.eff_count >= target_count
        });
    }
    return status;
}

std::tuple<Tensor, std::vector<Tensor>> EventGenerator::integrate_and_optimize(
    ChannelState& channel, TensorVec& events, bool run_optim
) {
    auto& weights = events.at(0);
    auto weights_cpu = weights.cpu();
    auto w_view = weights_cpu.view<double,1>();
    std::size_t sample_count = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) ++sample_count;
        channel.cross_section.push(w_view[i]);
    }
    channel.total_sample_count += sample_count; //w_view.size();

    if (run_optim) {
        if (channel.vegas_optimizer) {
            auto hist = channel.vegas_histogram->run({events.at(2), weights});
            channel.vegas_optimizer->add_data(hist.at(0), hist.at(1));
        }
        if (channel.discrete_optimizer) {
            TensorVec args{events.begin() + 3, events.end()};
            args.push_back(weights);
            auto hist = channel.discrete_histogram->run(args);
            channel.discrete_optimizer->add_data(hist);
        }
        if (channel.job_count == 0) {
            if (channel.vegas_optimizer) channel.vegas_optimizer->optimize();
            if (channel.discrete_optimizer) channel.discrete_optimizer->optimize();
            double rsd = channel.cross_section.rel_std_dev();
            if (rsd < _config.optimization_threshold * channel.best_rsd) {
                channel.iters_without_improvement = 0;
            } else {
                ++channel.iters_without_improvement;
                if (channel.iters_without_improvement >= _config.optimization_patience) {
                    channel.needs_optimization = false;
                }
            }
            channel.best_rsd = std::min(rsd, channel.best_rsd);
            ++channel.iterations;
        }
    }

    double total_mean = 0., total_var = 0.;
    std::size_t total_count = 0, total_integ_count = 0;
    std::size_t iterations = 0;
    bool optimized = true;
    for (auto& channel : _channels) {
        total_mean += channel.cross_section.mean();
        total_var += channel.cross_section.variance() / channel.cross_section.count();
        total_count += channel.total_sample_count;
        total_integ_count += channel.cross_section.count();
        iterations = std::max(channel.iterations, iterations);
        if (channel.needs_optimization) optimized = false;
    }
    _status_all.mean = total_mean;
    _status_all.error = std::sqrt(total_var);
    _status_all.rel_std_dev = std::sqrt(total_var * total_integ_count) / total_mean;
    _status_all.count = total_count;
    _status_all.count_integral = total_integ_count;
    _status_all.iterations = iterations;
    _status_all.optimized = optimized;
    for (auto& channel : _channels) {
        channel.integral_fraction = channel.cross_section.mean() / total_mean;
    }

    return {weights_cpu, events};
}

void EventGenerator::start_job(
    ChannelState& channel, std::size_t batch_size, std::size_t vegas_job_count
) {
    std::size_t new_job_id = _job_id;
    ++_job_id;
    ++channel.job_count;
    auto& job = std::get<0>(
        _running_jobs.emplace(new_job_id, RunningJob{channel.index, {}, vegas_job_count})
    )->second;
    default_thread_pool().submit([new_job_id, batch_size, &job, &channel] () {
        job.events = channel.runtime->run({Tensor({batch_size})});
        return new_job_id;
    });
}

void EventGenerator::start_vegas_jobs(ChannelState& channel) {
    std::size_t vegas_job_count =
        (channel.batch_size + _config.batch_size - 1) / _config.batch_size;
    for (std::size_t i = 0; i < vegas_job_count; ++i) {
        std::size_t batch_size = std::min(
            _config.batch_size, channel.batch_size - i * _config.batch_size
        );
        start_job(channel, batch_size, vegas_job_count);
    }
    channel.batch_size = std::min(channel.batch_size * 2, _config.max_batch_size);
}

void EventGenerator::clear_channel(ChannelState& channel) {
    channel.eff_count = 0;
    channel.max_weight = 0;
    channel.writer.clear();
    channel.cross_section.reset();
    channel.large_weights.clear();
}

void EventGenerator::update_max_weight(ChannelState& channel, Tensor weights) {
    if (channel.eff_count > _config.freeze_max_weight_after) return;

    auto w_view = weights.view<double,1>();
    double chan_max_weight = _max_weight * channel.integral_fraction;
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) w_min_nonzero = w;
        if (w > channel.max_weight) {
            channel.large_weights.push_back(w);
        }
    }
    if (channel.max_weight == 0) channel.max_weight = w_min_nonzero;
    std::sort(channel.large_weights.begin(), channel.large_weights.end(), std::greater{});

    double w_sum = 0, w_prev = 0;
    double max_truncation = _config.max_overweight_truncation * std::min(
        channel.integral_fraction * _config.target_count,
        static_cast<double>(_config.freeze_max_weight_after)
    );
    std::size_t count = 0;
    for (auto w : channel.large_weights) {
        if (w < channel.max_weight) break;
        w_sum += w;
        ++count;
        if (w_sum / w - count > max_truncation) {
            if (channel.max_weight < w) {
                channel.eff_count *= channel.max_weight / w_prev;
                channel.max_weight = w_prev;
            }
            break;
        }
        w_prev = w;
    }
    channel.large_weights.erase(
        channel.large_weights.begin() + count, channel.large_weights.end()
    );
}

void EventGenerator::unweight_and_write(
    ChannelState& channel, const std::vector<Tensor>& events
) {
    std::vector<Tensor> unweighter_args(events.begin(), events.begin() + 2);
    unweighter_args.push_back(Tensor(channel.max_weight, _context->device()));
    auto unw_events = _unweighter->run(unweighter_args);
    auto unw_weights = unw_events.at(0).cpu();
    auto w_view = unw_weights.view<double,1>();
    auto unw_momenta = unw_events.at(1).cpu();
    auto mom_view = unw_momenta.view<double,3>();

    EventBuffer buffer(channel.writer.particle_count());
    auto& buf_event = buffer.event();
    auto buf_particles = buffer.particles();
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        buf_event.weight = w_view[i];
        auto event_mom = mom_view[i];
        for (std::size_t j = 0; j < event_mom.size(); ++j) {
            auto particle_mom = event_mom[j];
            auto& buf_particle = buf_particles[j];
            buf_particle.e = particle_mom[0];
            buf_particle.px = particle_mom[1];
            buf_particle.py = particle_mom[2];
            buf_particle.pz = particle_mom[3];
        }
        channel.writer.write(buffer);
    }

    channel.eff_count += w_view.size();
    double total_eff_count = 0.;
    bool done = true;
    for (auto& channel : _channels) {
        double chan_target = channel.integral_fraction * _config.target_count;
        if (channel.eff_count < chan_target) {
            total_eff_count += channel.eff_count;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status_all.count_unweighted = total_eff_count;
    _status_all.done = done;
}

void EventGenerator::print_gen_init() {
    println(
        "┌ Integration and unweighting ────────────────────────────────────────────────────────────┐\n"
        "│ Result:                                                                                 │\n"
        "│ Rel. error:                                                                             │\n"
        "│ Rel. stddev:                                                                            │\n"
        "│ Number of events:                                                                       │\n"
        "│ Unweighting eff.:                                                                       │\n"
        "│ Unweighted events:                                                                      │\n"
        "└─────────────────────────────────────────────────────────────────────────────────────────┘\n"
    );

    if (_channels.size() > 1) {
        println(
            "┌ Individual channels ────────────────────────────────────────────────────────────────────┐\n"
            "│ #   integral ↓      RSD      uweff    N      opt    unweighted                          │"
        );
        for (std::size_t i = 0; i < _channels.size(); ++i) {
            if (i == 20) {
                println("│ ..{:<86}│", "");
                break;
            } else {
                println("│{:<89}│", i);
            }
        }
        println(
            "└─────────────────────────────────────────────────────────────────────────────────────────┘\n"
        );
    }
}

void EventGenerator::print_gen_update() {
    std::string int_str, rel_str, rsd_str, uweff_str;
    if (!std::isnan(_status_all.error)) {
        double rel_err = _status_all.error / _status_all.mean;
        int_str = format_with_error(_status_all.mean, _status_all.error);
        rel_str = std::format("{:.4f} %", rel_err * 100);
        rsd_str = std::format("{:.3f}", _status_all.rel_std_dev);
        uweff_str = std::format(
            "{:.5f}", _status_all.count_unweighted / _status_all.count_integral
        );
    }
    std::string unw_str = std::format(
        "{} / {}",
        format_si_prefix(_status_all.count_unweighted),
        format_si_prefix(_status_all.count_target)
    );
    if (!_status_all.done) {
        unw_str = std::format(
            "{:<15} {}",
            unw_str,
            format_progress(_status_all.count_unweighted / _status_all.count_target, 52)
        );
    }
    madevent::print(
        "\0337\033[{}F" // save cursor position, go up {} lines
        "\033[2K│ Result:            {:<69}│\n"
        "\033[2K│ Rel. error:        {:<69}│\n"
        "\033[2K│ Rel. stddev:       {:<69}│\n"
        "\033[2K│ Unweighting eff.:  {:<69}│\n"
        "\033[2K│ Number of events:  {:<69}│\n"
        "\033[2K│ Unweighted events: {:<69}│\n",
        _channels.size() == 1 ? 8 : (
            _channels.size() > 20 ? 33 : _channels.size() + 12
        ),
        int_str,
        rel_str,
        rsd_str,
        uweff_str,
        format_si_prefix(_status_all.count),
        unw_str
    );

    if (_channels.size() > 1) {
        auto channels = channel_status();
        std::sort(
            channels.begin(),
            channels.end(),
            [] (auto& chan1, auto& chan2) {
                return chan1.mean > chan2.mean;
            }
        );

        print("\n\n\n\n");
        for (auto& channel : channels | std::views::take(20)) {
            std::string int_str, rsd_str, count_str, unw_str, opt_str;
            if (!std::isnan(channel.error)) {
                int_str = format_with_error(channel.mean, channel.error);
                rsd_str = std::format("{:.3f}", channel.rel_std_dev);
                uweff_str = std::format(
                    "{:.5f}", channel.count_unweighted / channel.count_integral
                );
                count_str = format_si_prefix(channel.count);
                opt_str = std::format(
                    "{} {}", channel.iterations, channel.optimized || channel.done ? "✓" : ""
                );
                std::string unw_count_str = std::format(
                    "{:>5} / {:>5}",
                    format_si_prefix(channel.count_unweighted),
                    format_si_prefix(channel.count_target)
                );
                std::string progress;
                if (!_status_all.done) {
                    progress = format_progress(channel.count_unweighted / channel.count_target, 19);
                }
                unw_str = std::format("{:<15} {:<19}", unw_count_str, progress);
            }
            madevent::println(
                "\033[2K│ {:<4}{:<16}{:<9}{:<9}{:<7}{:<7}{} │",
                channel.index, int_str, rsd_str, uweff_str, count_str, opt_str, unw_str
            );
        }
    }

    // restore cursor position
    print("\0338");
    std::cout << std::flush;
}
