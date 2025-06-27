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
        context
    )),
    _status_all({0, 0., 0., 0., 0, 0., static_cast<double>(config.target_count), false}),
    _writer(file_name, channels.at(0).particle_count())
{
    std::size_t i = 0;
    fs::path file_path(file_name);
    fs::path temp_path = temp_file_dir.value_or(file_path.parent_path());
    for (auto& channel : channels) {
        if (channel.flags() != integrand_flags) {
            throw std::invalid_argument(
                "Integrand flags must be sample | return_momenta | return_random"
            );
        }
        auto chan_path = temp_path / file_path.stem();
        chan_path += std::format(".channel{}.npy", i);
        std::optional<VegasGridOptimizer> vegas_optimizer;
        if (const auto& name = channel.vegas_grid_name(); name) {
            vegas_optimizer = VegasGridOptimizer(context, *name, config.vegas_damping);
        }
        _channels.push_back({
            i,
            build_runtime(channel.function(), context),
            EventFile(chan_path.string(), channel.particle_count(), EventFile::create, true),
            vegas_optimizer,
            config.start_batch_size
        });
        ++i;
    }
}

void EventGenerator::survey() {
    bool done = false;
    for (std::size_t iter = 0; !done && iter < _config.survey_max_iters; ++iter) {
        done = true;
        for (auto& channel : _channels) {
            _abort_check_function();
            //clear_channel(channel);
            auto [weights, events] = generate_channel(channel, true);

            if (iter >= _config.survey_min_iters - 1 && (
                iter == _config.survey_max_iters - 1 ||
                channel.cross_section.rel_error() < _config.survey_target_precision
            )) {
                update_max_weight(channel, weights);
                unweight_and_write(channel, events);
            } else {
                done = false;
            }
        }
    }
}

void EventGenerator::generate() {
    print_gen_init();
    while (!_status_all.done) {
        for (auto& channel : _channels) {
            _abort_check_function();
            if (channel.eff_count >= channel.integral_fraction * _config.target_count) {
                continue;
            }
            auto [weights, events] = generate_channel(channel, false);
            update_max_weight(channel, weights);
            unweight_and_write(channel, events);
            print_gen_update();
        }
        if (_status_all.done) unweight_all();
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
            channel.eff_count,
            target_count,
            channel.iterations,
            channel.eff_count >= target_count
        });
    }
    return status;
}

std::tuple<Tensor, std::vector<Tensor>> EventGenerator::generate_channel(
    ChannelState& channel, bool always_optimize
) {
    bool run_optim = channel.vegas_optimizer && (
        channel.needs_optimization || always_optimize
    );
    if (run_optim) clear_channel(channel);

    auto events = channel.runtime->run({Tensor({channel.batch_size})});
    channel.batch_size = std::min(channel.batch_size * 2, _config.max_batch_size);

    auto weights = events.at(0).cpu();
    auto w_view = weights.view<double,1>();
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        channel.cross_section.push(w_view[i]);
    }
    channel.total_sample_count += w_view.size();

    if (run_optim) {
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
        channel.vegas_optimizer->optimize(weights, events.at(2));
        ++channel.iterations;
    }

    double total_mean = 0., total_var = 0.;
    std::size_t total_count = 0, total_integ_count = 0;
    std::size_t iterations = 0;
    for (auto& channel : _channels) {
        total_mean += channel.cross_section.mean();
        total_var += channel.cross_section.variance() / channel.cross_section.count();
        total_count += channel.total_sample_count;
        total_integ_count += channel.cross_section.count();
        iterations = std::max(channel.iterations, iterations);
    }
    _status_all.mean = total_mean;
    _status_all.error = std::sqrt(total_var);
    _status_all.rel_std_dev = std::sqrt(total_var * total_integ_count) / total_mean;
    _status_all.count = total_count;
    _status_all.iterations = iterations;
    for (auto& channel : _channels) {
        channel.integral_fraction = channel.cross_section.mean() / total_mean;
    }

    return {weights, events};
}

void EventGenerator::clear_channel(ChannelState& channel) {
    channel.eff_count = 0;
    channel.max_weight = 0;
    channel.writer.clear();
    channel.cross_section.reset();
    std::erase_if(
        _large_weights,
        [&channel](auto item) { return std::get<1>(item) == channel.index; }
    );
}

void EventGenerator::update_max_weight(ChannelState& channel, Tensor weights) {
    auto w_view = weights.view<double,1>();
    double chan_max_weight = _max_weight * channel.integral_fraction;
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) w_min_nonzero = w;
        if (w > channel.max_weight) {
            _large_weights.push_back({w, channel.index});
        }
    }
    if (channel.max_weight == 0) channel.max_weight = w_min_nonzero;

    std::sort(
        _large_weights.begin(), _large_weights.end(),
        [this](auto& item1, auto& item2) {
            auto [w1, id1] = item1;
            auto [w2, id2] = item2;
            return w1 / _channels.at(id1).max_weight > w2 / _channels.at(id2).max_weight;
        }
    );

    double w_sum = 0, w_prev = 0;
    double max_truncation =
        _config.max_overweight_truncation * channel.integral_fraction * _config.target_count;
    std::size_t count = 0;
    for (auto [w, chan_index] : _large_weights) {
        if (chan_index != channel.index) continue;
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

    std::size_t max_overweights = _config.max_overweight_fraction * _config.target_count;
    if (_large_weights.size() > max_overweights) {
        _large_weights.erase(_large_weights.begin() + max_overweights, _large_weights.end());
        std::vector<bool> found_channels(_channels.size());
        for (auto [w, chan_index] : std::views::reverse(_large_weights)) {
            if (found_channels.at(chan_index)) continue;
            found_channels.at(chan_index) = true;
            auto& chan = _channels.at(chan_index);
            if (chan.max_weight < w) {
                chan.eff_count *= chan.max_weight / w;
                chan.max_weight = w;
            }
        }
    }
}

void EventGenerator::unweight_and_write(
    ChannelState& channel, const std::vector<Tensor>& events
) {
    std::vector<Tensor> unweighter_args(events.begin(), events.begin() + 2);
    unweighter_args.push_back(Tensor(channel.max_weight, _context->device()));
    auto unw_events = _unweighter->run(unweighter_args);
    auto unw_weights = unw_events.at(0);
    auto w_view = unw_weights.view<double,1>();
    auto unw_momenta = unw_events.at(1);
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
        "Integration and unweighting\n"
        "--------------------------------------------------------------------------\n"
        " Result:\n"
        " Rel. error:\n"
        " Rel. stddev:\n"
        " Number of events:\n"
        " Unweighted events:\n"
        "--------------------------------------------------------------------------\n"
    );

    if (_channels.size() > 1) {
        println(
            "Individual channels: \n"
            "--------------------------------------------------------------------------\n"
            " #   integral ↓      RSD      N      unweighted"
        );
        for (std::size_t i = 0; i < _channels.size(); ++i) {
            if (i == 20) {
                println(" ..");
                break;
            } else {
                println(" {}", i);
            }
        }
        println(
            "--------------------------------------------------------------------------\n"
        );
    }
}

void EventGenerator::print_gen_update() {
    std::string int_str, rel_str, rsd_str;
    if (!std::isnan(_status_all.error)) {
        double rel_err = _status_all.error / _status_all.mean;
        int_str = format_with_error(_status_all.mean, _status_all.error);
        rel_str = std::format("{:.4f} %", rel_err * 100);
        rsd_str = std::format("{:.3f}", _status_all.rel_std_dev);
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
            format_progress(_status_all.count_unweighted / _status_all.count_target, 36)
        );
    }
    madevent::print(
        "\0337\033[{}F" // save cursor position, go up {} lines
        "\033[2K Result:            {}\n"
        "\033[2K Rel. error:        {}\n"
        "\033[2K Rel. stddev:       {}\n"
        "\033[2K Number of events:  {}\n"
        "\033[2K Unweighted events: {}\n",
        _channels.size() == 1 ? 7 : (
            _channels.size() > 20 ? 33 : _channels.size() + 12
        ),
        int_str,
        rel_str,
        rsd_str,
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

        print("\n\n\n\n\n");
        for (auto& channel : channels | std::views::take(20)) {
            std::string int_str, rsd_str, count_str, unw_str;
            if (!std::isnan(channel.error)) {
                int_str = format_with_error(channel.mean, channel.error);
                rsd_str = std::format("{:.3f}", channel.rel_std_dev);
                count_str = format_si_prefix(channel.count);
                unw_str = std::format(
                    "{} / {}",
                    format_si_prefix(channel.count_unweighted),
                    format_si_prefix(channel.count_target)
                );
                if (!_status_all.done) {
                    unw_str = std::format(
                        "{:<15} {}",
                        unw_str,
                        format_progress(channel.count_unweighted / channel.count_target, 19)
                    );
                }
            }
            madevent::println(
                "\033[2K {:<4}{:<16}{:<9}{:<7}{}",
                channel.index,
                int_str,
                rsd_str,
                count_str,
                unw_str
            );
        }
    }

    // restore cursor position
    print("\0338");
    std::cout << std::flush;
}
