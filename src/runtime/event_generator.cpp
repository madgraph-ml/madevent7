#include "madevent/runtime/event_generator.h"

#include <cmath>
#include <format>
#include <ranges>

#include "madevent/util.h"

using namespace madevent;

const EventGenerator::Config EventGenerator::default_config = {};

EventGenerator::EventGenerator(
    ContextPtr context,
    const std::vector<Integrand>& channels,
    const std::string& temp_file_prefix,
    const Config& config
) :
    _context(context),
    _config(config),
    _unweighter(build_runtime(
        Unweighter({channels.at(0).return_types().at(0),
                    channels.at(0).return_types().at(1)})
            .function(),
        context,
        false
    )),
    _status_all(
        {0,
         0.,
         0.,
         0.,
         0,
         0,
         0.,
         static_cast<double>(config.target_count),
         false,
         false}
    ),
    _job_id(0) {
    std::size_t i = 0;
    for (auto& channel : channels) {
        if (channel.flags() != integrand_flags) {
            throw std::invalid_argument(
                "Integrand flags must be sample | return_momenta | return_random | "
                "return_discrete"
            );
        }
        std::optional<VegasGridOptimizer> vegas_optimizer;
        RuntimePtr vegas_histogram = nullptr;
        if (const auto& name = channel.vegas_grid_name(); name) {
            vegas_optimizer = VegasGridOptimizer(context, *name, config.vegas_damping);
            VegasHistogram hist(channel.vegas_dimension(), channel.vegas_bin_count());
            vegas_histogram = build_runtime(hist.function(), context, false);
        }
        std::vector<std::string> prob_names;
        std::vector<std::size_t> option_counts;
        auto add_names = Overloaded{
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
            .index = i,
            .runtime = build_runtime(channel.function(), context, false),
            .event_file = EventFile(
                std::format("{}_channel{}_events.npy", temp_file_prefix, i),
                DataLayout::of<EventIndicesRecord, ParticleRecord>(),
                channel.particle_count(),
                EventFile::create,
                false
            ),
            .weight_file = EventFile(
                std::format("{}_channel{}_weights.npy", temp_file_prefix, i),
                DataLayout::of<EventWeightRecord, EmptyParticleRecord>(),
                0,
                EventFile::create,
                true
            ),
            .vegas_optimizer = vegas_optimizer,
            .vegas_histogram = std::move(vegas_histogram),
            .discrete_optimizer = discrete_optimizer,
            .discrete_histogram = std::move(discrete_histogram),
            .batch_size = config.start_batch_size,
        });
        ++i;
    }
}

void EventGenerator::survey() {
    reset_start_time();
    bool done = false;
    auto& thread_pool = default_thread_pool();
    std::size_t min_iters = _config.survey_min_iters;
    std::size_t max_iters = std::max(min_iters, _config.survey_max_iters);
    double target_precision = _config.survey_target_precision;

    for (std::size_t iter = 0; !done && iter < max_iters; ++iter) {
        for (auto& channel : _channels) {
            if (iter >= min_iters &&
                channel.cross_section.rel_error() < target_precision) {
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
            auto [weights, events] =
                integrate_and_optimize(channel, job.events, run_optim);

            if (iter >= min_iters - 1) {
                update_max_weight(channel, weights);
                unweight_and_write(channel, events);
                if (channel.job_count == 0 &&
                    channel.cross_section.rel_error() < target_precision) {
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
    reset_start_time();
    print_gen_init();
    auto& thread_pool = default_thread_pool();
    std::size_t channel_index = 0;
    std::size_t target_job_count = 2 * thread_pool.thread_count();
    while (true) {
        _abort_check_function();

        std::size_t job_count_before;
        do {
            job_count_before = _running_jobs.size();
            for (std::size_t i = 0;
                 i < _channels.size() && _running_jobs.size() < target_job_count;
                 ++i, channel_index = (channel_index + 1) % _channels.size()) {
                auto& channel = _channels.at(channel_index);
                if (channel.eff_count >=
                    channel.integral_fraction * _config.target_count) {
                    continue;
                }
                if ((channel.vegas_optimizer || channel.discrete_optimizer) &&
                    channel.needs_optimization) {
                    if (channel.job_count == 0) {
                        start_vegas_jobs(channel);
                    }
                } else {
                    start_job(channel, _config.batch_size);
                }
            }
        } while (_running_jobs.size() - job_count_before > 0);

        if (auto job_id = thread_pool.wait()) {
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            bool run_optim = (channel.vegas_optimizer || channel.discrete_optimizer) &&
                channel.needs_optimization;
            if (run_optim && channel.job_count == job.vegas_job_count) {
                clear_channel(channel);
            }
            --channel.job_count;

            auto [weights, events] =
                integrate_and_optimize(channel, job.events, run_optim);
            update_max_weight(channel, weights);
            unweight_and_write(channel, events);
            print_gen_update();
            _running_jobs.erase(*job_id);
        } else {
            if (_status_all.done) {
                unweight_all();
            }
            if (_status_all.done) {
                break;
            }
        }
    }
}

void EventGenerator::combine_to_compact_npy(const std::string& file_name) {
    reset_start_time();
    auto [channel_data, particle_count, norm_factor] = init_combine();
    EventBuffer buffer(
        0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()
    );
    EventFile event_file(
        file_name,
        DataLayout::of<EventFullRecord, ParticleRecord>(),
        particle_count,
        EventFile::create
    );
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    print_combine_init();
    while (true) {
        read_and_combine(channel_data, buffer, norm_factor);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        if (event_count - last_update_count > 100000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
        event_file.write(buffer);
    }
    print_combine_update(event_count);
}

void EventGenerator::combine_to_lhe_npy(
    const std::string& file_name, const LHECompleter& lhe_completer
) {
    reset_start_time();
    auto [channel_data, particle_count, norm_factor] = init_combine();
    EventBuffer buffer(
        0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()
    );
    EventBuffer buffer_out(
        0,
        lhe_completer.max_particle_count(),
        DataLayout::of<PackedLHEEvent, PackedLHEParticle>()
    );
    EventFile event_file(
        file_name,
        DataLayout::of<PackedLHEEvent, PackedLHEParticle>(),
        lhe_completer.max_particle_count(),
        EventFile::create
    );
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    LHEEvent lhe_event;
    print_combine_init();
    while (true) {
        read_and_combine(channel_data, buffer, norm_factor);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            fill_lhe_event(lhe_completer, lhe_event, buffer, i);
            buffer_out.event<PackedLHEEvent>(i).from_lhe_event(lhe_event);
            std::size_t j = 0;
            for (; j < lhe_event.particles.size(); ++j) {
                buffer_out.particle<PackedLHEParticle>(i, j).from_lhe_particle(
                    lhe_event.particles[j]
                );
            }
            for (; j < lhe_completer.max_particle_count(); ++j) {
                buffer_out.particle<PackedLHEParticle>(i, j).from_lhe_particle(
                    LHEParticle{}
                );
            }
        }
        event_file.write(buffer_out);
        if (event_count - last_update_count > 100000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
    }
    print_combine_update(event_count);
}

void EventGenerator::combine_to_lhe(
    const std::string& file_name, const LHECompleter& lhe_completer
) {
    reset_start_time();
    auto [channel_data, particle_count, norm_factor] = init_combine();
    EventBuffer buffer(
        0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()
    );
    LHEFileWriter event_file(file_name, LHEMeta{});
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    LHEEvent lhe_event;
    print_combine_init();
    while (true) {
        read_and_combine(channel_data, buffer, norm_factor);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            fill_lhe_event(lhe_completer, lhe_event, buffer, i);
            event_file.write(lhe_event);
        }
        if (event_count - last_update_count > 100000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
    }
    print_combine_update(event_count);
}

void EventGenerator::reset_start_time() {
    _start_time = std::chrono::steady_clock::now();
}

void EventGenerator::unweight_all() {
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    bool done = true;
    double total_eff_count = 0.;
    for (auto& channel : _channels) {
        unweight_channel(channel, rand_gen);

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

void EventGenerator::unweight_channel(ChannelState& channel, std::mt19937 rand_gen) {
    std::size_t buf_size = 1000000;
    std::uniform_real_distribution<double> rand_dist;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    std::size_t accept_count = 0;
    for (std::size_t i = 0; i < channel.weight_file.event_count(); i += buf_size) {
        channel.weight_file.seek(i);
        channel.weight_file.read(buffer, buf_size);
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            auto weight = buffer.event<EventWeightRecord>(j).weight();
            if (weight / channel.max_weight < rand_dist(rand_gen)) {
                weight = 0;
            } else {
                weight = std::max(weight.value(), channel.max_weight);
                ++accept_count;
            }
        }
        channel.weight_file.seek(i);
        channel.weight_file.write(buffer);
    }
    channel.eff_count = accept_count;
}

double
EventGenerator::channel_weight_sum(ChannelState& channel, std::size_t event_count) {
    std::size_t buf_size = 1000000;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    double weight_sum = 0;
    channel.weight_file.seek(0);
    std::size_t unweighted_count = 0;
    for (std::size_t i = 0; i < channel.weight_file.event_count(); i += buf_size) {
        channel.weight_file.read(buffer, buf_size);
        bool done = false;
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            if (unweighted_count == event_count) {
                done = true;
                break;
            }
            double weight = buffer.event<EventWeightRecord>(j).weight();
            if (weight == 0.) {
                continue;
            }
            weight_sum += weight / channel.max_weight;
            ++unweighted_count;
        }
        if (done) {
            break;
        }
    }
    return weight_sum;
}

std::vector<EventGenerator::Status> EventGenerator::channel_status() const {
    std::vector<EventGenerator::Status> status;
    for (auto& channel : _channels) {
        double target_count = channel.integral_fraction * _config.target_count;
        status.push_back(
            {channel.index,
             channel.cross_section.mean(),
             channel.cross_section.error(),
             channel.cross_section.rel_std_dev(),
             channel.total_sample_count,
             channel.cross_section.count(),
             channel.eff_count,
             target_count,
             channel.iterations,
             !channel.needs_optimization,
             channel.eff_count >= target_count}
        );
    }
    return status;
}

std::tuple<Tensor, std::vector<Tensor>> EventGenerator::integrate_and_optimize(
    ChannelState& channel, TensorVec& events, bool run_optim
) {
    auto& weights = events.at(0);
    auto weights_cpu = weights.cpu();
    auto w_view = weights_cpu.view<double, 1>();
    std::size_t sample_count = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) {
            ++sample_count;
        }
        channel.cross_section.push(w_view[i]);
    }
    channel.total_sample_count += sample_count; // w_view.size();

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
            if (channel.vegas_optimizer) {
                channel.vegas_optimizer->optimize();
            }
            if (channel.discrete_optimizer) {
                channel.discrete_optimizer->optimize();
            }
            double rsd = channel.cross_section.rel_std_dev();
            if (rsd < _config.optimization_threshold * channel.best_rsd) {
                channel.iters_without_improvement = 0;
            } else {
                ++channel.iters_without_improvement;
                if (channel.iters_without_improvement >=
                    _config.optimization_patience) {
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
        if (channel.needs_optimization) {
            optimized = false;
        }
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
    auto& job =
        std::get<0>(_running_jobs.emplace(
                        new_job_id, RunningJob{channel.index, {}, vegas_job_count}
                    ))
            ->second;
    default_thread_pool().submit([new_job_id, batch_size, &job, &channel]() {
        job.events = channel.runtime->run({Tensor({batch_size})});
        return new_job_id;
    });
}

void EventGenerator::start_vegas_jobs(ChannelState& channel) {
    std::size_t vegas_job_count =
        (channel.batch_size + _config.batch_size - 1) / _config.batch_size;
    for (std::size_t i = 0; i < vegas_job_count; ++i) {
        std::size_t batch_size =
            std::min(_config.batch_size, channel.batch_size - i * _config.batch_size);
        start_job(channel, batch_size, vegas_job_count);
    }
    channel.batch_size = std::min(channel.batch_size * 2, _config.max_batch_size);
}

void EventGenerator::clear_channel(ChannelState& channel) {
    channel.eff_count = 0;
    channel.max_weight = 0;
    channel.event_file.clear();
    channel.weight_file.clear();
    channel.cross_section.reset();
    channel.large_weights.clear();
}

void EventGenerator::update_max_weight(ChannelState& channel, Tensor weights) {
    if (channel.eff_count > _config.freeze_max_weight_after) {
        return;
    }

    auto w_view = weights.view<double, 1>();
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) {
            w_min_nonzero = w;
        }
        if (w > channel.max_weight) {
            channel.large_weights.push_back(w);
        }
    }
    if (channel.max_weight == 0) {
        channel.max_weight = w_min_nonzero;
    }
    std::sort(
        channel.large_weights.begin(), channel.large_weights.end(), std::greater{}
    );

    double w_sum = 0, w_prev = 0;
    double max_truncation = _config.max_overweight_truncation *
        std::min(channel.integral_fraction * _config.target_count,
                 static_cast<double>(_config.freeze_max_weight_after));
    std::size_t count = 0;
    for (auto w : channel.large_weights) {
        if (w < channel.max_weight) {
            break;
        }
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
    auto w_view = unw_weights.view<double, 1>();
    auto unw_momenta = unw_events.at(1).cpu();
    auto mom_view = unw_momenta.view<double, 3>();

    EventBuffer event_buffer(
        w_view.size(),
        channel.event_file.particle_count(),
        DataLayout::of<EventIndicesRecord, ParticleRecord>()
    );
    EventBuffer weight_buffer(
        w_view.size(), 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>()
    );
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        weight_buffer.event<EventWeightRecord>(i).weight() = w_view[i];
        auto event = event_buffer.event<EventIndicesRecord>(i);
        event.diagram_index() = 0;
        event.color_index() = 0;
        event.flavor_index() = 0;
        event.helicity_index() = 0;
        auto event_mom = mom_view[i];
        for (std::size_t j = 0; j < event_mom.size(); ++j) {
            auto particle_mom = event_mom[j];
            auto particle = event_buffer.particle<ParticleRecord>(i, j);
            particle.energy() = particle_mom[0];
            particle.px() = particle_mom[1];
            particle.py() = particle_mom[2];
            particle.pz() = particle_mom[3];
        }
    }
    channel.event_file.write(event_buffer);
    channel.weight_file.write(weight_buffer);

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

std::tuple<std::vector<EventGenerator::CombineChannelData>, std::size_t, double>
EventGenerator::init_combine() {
    std::vector<EventGenerator::CombineChannelData> channel_data;
    std::size_t count_sum = 0;
    std::size_t particle_count = 0;
    double weight_sum = 0.;
    for (auto& channel : _channels) {
        particle_count = std::max(particle_count, channel.event_file.particle_count());
        std::size_t count =
            std::round(channel.integral_fraction * _config.target_count);
        count_sum += count;
        channel.event_file.seek(0);
        weight_sum += channel_weight_sum(channel, count);
        channel.weight_file.seek(0);
        channel_data.push_back({
            .cum_count = count_sum,
            .event_buffer = EventBuffer(
                0,
                channel.event_file.particle_count(),
                DataLayout::of<EventIndicesRecord, ParticleRecord>()
            ),
            .weight_buffer = EventBuffer(
                0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>()
            ),
            .buffer_index = 0,
        });
    }
    return {channel_data, particle_count, _status_all.mean * count_sum / weight_sum};
}

void EventGenerator::read_and_combine(
    std::vector<EventGenerator::CombineChannelData>& channel_data,
    EventBuffer& buffer,
    double norm_factor
) {
    std::size_t batch_size = 1000;
    std::size_t event_count = std::min(batch_size, channel_data.back().cum_count);
    buffer.resize(event_count);

    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    for (std::size_t event_index = 0; event_index < event_count; ++event_index) {
        std::size_t random_index = std::uniform_int_distribution<
            std::size_t>(0, channel_data.back().cum_count - 1)(rand_gen);
        auto sampled_chan = std::lower_bound(
            channel_data.begin(),
            channel_data.end(),
            random_index,
            [](auto& chan, std::size_t val) { return chan.cum_count < val; }
        );
        std::for_each(sampled_chan, channel_data.end(), [](auto& chan) {
            --chan.cum_count;
        });
        auto& channel = _channels.at(sampled_chan - channel_data.begin());

        double weight = 0.;
        while (true) {
            if (sampled_chan->buffer_index ==
                sampled_chan->event_buffer.event_count()) {
                channel.event_file.read(sampled_chan->event_buffer, batch_size);
                channel.weight_file.read(sampled_chan->weight_buffer, batch_size);
                sampled_chan->buffer_index = 0;
            }
            weight = sampled_chan->weight_buffer
                         .event<EventWeightRecord>(sampled_chan->buffer_index)
                         .weight();
            if (weight != 0.) {
                break;
            }
            ++sampled_chan->buffer_index;
        }

        auto event_in =
            sampled_chan->event_buffer.event<EventIndicesRecord>(event_index);
        auto event_out = buffer.event<EventFullRecord>(event_index);
        event_out.weight() = std::max(1., weight / channel.max_weight) * norm_factor;
        event_out.diagram_index() = event_in.diagram_index();
        event_out.color_index() = event_in.color_index();
        event_out.flavor_index() = event_in.flavor_index();
        event_out.helicity_index() = event_in.helicity_index();

        std::size_t i = 0;
        for (; i < sampled_chan->event_buffer.particle_count(); ++i) {
            auto particle_in =
                sampled_chan->event_buffer.particle<ParticleRecord>(event_index, i);
            auto particle_out = buffer.particle<ParticleRecord>(event_index, i);
            particle_out.energy() = particle_in.energy();
            particle_out.px() = particle_in.px();
            particle_out.py() = particle_in.py();
            particle_out.pz() = particle_in.pz();
        }
        for (; i < buffer.particle_count(); ++i) {
            auto particle_out = buffer.particle<ParticleRecord>(event_index, i);
            particle_out.energy() = 0.;
            particle_out.px() = 0.;
            particle_out.py() = 0.;
            particle_out.pz() = 0.;
        }
        ++sampled_chan->buffer_index;
    }
}

void EventGenerator::fill_lhe_event(
    const LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index
) {
    EventRecord event_in = buffer.event<EventFullRecord>(event_index);
    lhe_event.particles.clear();
    for (std::size_t i = 0; i < buffer.particle_count(); ++i) {
        auto particle_in = buffer.particle<ParticleRecord>(event_index, i);
        if (particle_in.energy() == 0.) {
            break;
        }
        lhe_event.particles.push_back(
            LHEParticle{
                .px = particle_in.px(),
                .py = particle_in.py(),
                .pz = particle_in.pz(),
                .energy = particle_in.energy(),
            }
        );
    }
}

void EventGenerator::print_gen_init() {
    _last_print_time = std::chrono::steady_clock::now();

    std::size_t offset = 0;
    if (_channels.size() > 1) {
        _pretty_box_lower = PrettyBox(
            "Individual channels",
            _channels.size() < 21 ? _channels.size() + 1 : 22,
            {4, 16, 9, 9, 7, 7, 0}
        );
        _pretty_box_lower.set_row(
            0, {"#", "integral ↓", "RSD", "uweff", "N", "opt", "unweighted"}
        );
        if (_channels.size() > 20) {
            _pretty_box_lower.set_cell(21, 0, "..");
        }
        offset = _pretty_box_lower.line_count();
    }
    _pretty_box_upper = PrettyBox("Integration and unweighting", 7, {19, 0}, offset);
    _pretty_box_upper.set_column(
        0,
        {"Result:",
         "Rel. error:",
         "Rel. stddev:",
         "Number of events:",
         "Unweighting eff.:",
         "Unweighted events:",
         "Run time:"}
    );
    _pretty_box_upper.print_first();
    if (_channels.size() > 1) {
        _pretty_box_lower.print_first();
    }
}

void EventGenerator::print_gen_update() {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 0.1s && !_status_all.done) {
        return;
    }
    _last_print_time = std::chrono::steady_clock::now();

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
    std::string time_str = std::format(
        "{:%H:%M:%S}", std::chrono::round<std::chrono::seconds>(now - _start_time)
    );
    if (!_status_all.done) {
        unw_str = std::format(
            "{:<15} {}",
            unw_str,
            format_progress(_status_all.count_unweighted / _status_all.count_target, 52)
        );
    }
    _pretty_box_upper.set_column(
        1,
        {int_str,
         rel_str,
         rsd_str,
         uweff_str,
         format_si_prefix(_status_all.count),
         unw_str,
         time_str}
    );
    _pretty_box_upper.print_update();

    if (_channels.size() > 1) {
        auto channels = channel_status();
        std::sort(channels.begin(), channels.end(), [](auto& chan1, auto& chan2) {
            return chan1.mean > chan2.mean;
        });

        for (std::size_t row = 1; auto& channel : channels | std::views::take(20)) {
            std::string index_str = std::format("{}", channel.index);
            std::string int_str, rsd_str, count_str, unw_str, opt_str;
            if (!std::isnan(channel.error)) {
                int_str = format_with_error(channel.mean, channel.error);
                rsd_str = std::format("{:.3f}", channel.rel_std_dev);
                uweff_str = std::format(
                    "{:.5f}", channel.count_unweighted / channel.count_integral
                );
                count_str = format_si_prefix(channel.count);
                opt_str = std::format(
                    "{} {}",
                    channel.iterations,
                    channel.optimized || channel.done ? "✓" : ""
                );
                std::string unw_count_str = std::format(
                    "{:>5} / {:>5}",
                    format_si_prefix(channel.count_unweighted),
                    format_si_prefix(channel.count_target)
                );
                std::string progress;
                if (!_status_all.done) {
                    progress = format_progress(
                        channel.count_unweighted / channel.count_target, 19
                    );
                }
                unw_str = std::format("{:<15} {:<19}", unw_count_str, progress);
            }
            _pretty_box_lower.set_row(
                row,
                {index_str, int_str, rsd_str, uweff_str, count_str, opt_str, unw_str}
            );
            ++row;
        }
        _pretty_box_lower.print_update();
    }
}

void EventGenerator::print_combine_init() {
    _pretty_box_upper = PrettyBox("Writing final output", 2, {10, 0});
    _pretty_box_upper.set_column(0, {"Events:", "Run time:"});
    _pretty_box_upper.print_first();
}

void EventGenerator::print_combine_update(std::size_t count) {
    _pretty_box_upper.set_column(
        1,
        {std::format(
             "{:>5} / {:>5}   {}",
             format_si_prefix(count),
             format_si_prefix(_config.target_count),
             format_progress(static_cast<double>(count) / _config.target_count, 60)
         ),
         std::format(
             "{:%H:%M:%S}",
             std::chrono::round<std::chrono::seconds>(
                 std::chrono::steady_clock::now() - _start_time
             )
         )}
    );
    _pretty_box_upper.print_update();
}
