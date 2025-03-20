#pragma once

#include "madevent/phasespace/phasespace.h"

namespace madevent {

class DifferentialCrossSection : public FunctionGenerator {
public:
    using PidOptions = std::tuple<std::vector<int>, std::size_t>;

    DifferentialCrossSection(
        const std::vector<PidOptions>& pid_options,
        double e_cm2,
        double q2,
        std::size_t channel_count = 1,
        const std::vector<int64_t>& amp2_remap = {}
    ) :
        FunctionGenerator(
            {
                batch_four_vec_array(std::get<0>(pid_options.at(0)).size()),
                batch_float,
                batch_float
            },
            channel_count == 1 ?
                TypeVec{batch_float} :
                TypeVec{batch_float, batch_float_array(channel_count)}
        ),
        _pid_options(pid_options),
        _e_cm2(e_cm2),
        _q2(q2),
        _channel_count(channel_count),
        _amp2_remap(amp2_remap)
    {}

    const std::vector<PidOptions>& pid_options() const { return _pid_options; }
private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<PidOptions> _pid_options;
    double _e_cm2;
    double _q2;
    int64_t _channel_count;
    std::vector<int64_t> _amp2_remap;
};

class Unweighter : public FunctionGenerator {
public:
    Unweighter(std::size_t particle_count) :
        FunctionGenerator(
            {
                batch_four_vec_array(particle_count),
                batch_float, batch_float, batch_float, single_float
            },
            {
                batch_four_vec_array(particle_count),
                batch_float, batch_float, batch_float
            }
        )
    {}

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
};

class Integrand : public FunctionGenerator {
public:
    Integrand(
        const PhaseSpaceMapping& mapping,
        const DifferentialCrossSection& diff_xs,
        bool sample = false,
        bool unweight = false
    ) :
        FunctionGenerator(
            [&] {
                TypeVec arg_types;
                if (sample) {
                    arg_types.push_back(Type({batch_size}));
                } else {
                    arg_types.push_back(batch_float_array(mapping.random_dim()));
                }
                if (unweight) {
                    arg_types.push_back(single_float);
                }
                return arg_types;
            }(),
            {
                batch_four_vec_array(mapping.particle_count()),
                batch_float,
                batch_float,
                batch_float
            }
        ),
        _mapping(mapping),
        _diff_xs(diff_xs),
        _sample(sample),
        _unweight(unweight)
    {}
    std::size_t particle_count() const { return _mapping.particle_count(); }

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    bool _sample;
    bool _unweight;
};

}
