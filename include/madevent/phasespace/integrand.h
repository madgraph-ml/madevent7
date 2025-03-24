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
    Unweighter(const TypeVec& types, std::size_t particle_count) :
        FunctionGenerator(
            [&] {
                auto arg_types = types;
                arg_types.push_back(single_float);
                return arg_types;
            }(),
            types
        )
    {}

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
};

class Integrand : public FunctionGenerator {
public:
    inline static const int sample = 1;
    inline static const int unweight = 2;
    inline static const int return_momenta = 4;
    inline static const int return_x1_x2 = 8;
    inline static const int return_random = 16;

    Integrand(
        const PhaseSpaceMapping& mapping,
        const DifferentialCrossSection& diff_xs,
        int flags = 0
    ) :
        FunctionGenerator(
            [&] {
                TypeVec arg_types;
                if (flags & sample) {
                    arg_types.push_back(Type({batch_size}));
                } else {
                    arg_types.push_back(batch_float_array(mapping.random_dim()));
                }
                if (flags & unweight) arg_types.push_back(single_float);
                return arg_types;
            }(),
            [&] {
                TypeVec ret_types {batch_float};
                if (flags & return_momenta) {
                    ret_types.push_back(batch_four_vec_array(mapping.particle_count()));
                }
                if (flags & return_x1_x2) {
                    ret_types.push_back(batch_float);
                    ret_types.push_back(batch_float);
                }
                if (flags & return_random) {
                    ret_types.push_back(batch_float_array(mapping.random_dim()));
                }
                return ret_types;
            }()
        ),
        _mapping(mapping),
        _diff_xs(diff_xs),
        _flags(flags)
    {}
    std::size_t particle_count() const { return _mapping.particle_count(); }
    int flags() const { return _flags; }

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    int _flags;
};

}
