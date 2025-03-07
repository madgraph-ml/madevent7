#pragma once

#include "madevent/phasespace/phasespace.h"

namespace madevent {

class DifferentialCrossSection : public FunctionGenerator {
public:
    using PidOptions = std::tuple<std::vector<int>, std::size_t>;

    DifferentialCrossSection(
        const std::vector<PidOptions>& pid_options, double e_cm2, double q2
    ) :
        FunctionGenerator(
            {
                batch_four_vec_array(std::get<0>(pid_options.at(0)).size()),
                batch_float,
                batch_float
            },
            {batch_float}
        ),
        _pid_options(pid_options),
        _e_cm2(e_cm2),
        _q2(q2)
    {}

    const std::vector<PidOptions>& pid_options() const { return _pid_options; }
private:
    ValueList build_function_impl(FunctionBuilder& fb, const ValueList& args) const override;

    std::vector<PidOptions> _pid_options;
    double _e_cm2;
    double _q2;
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
                TypeList arg_types;
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

private:
    ValueList build_function_impl(FunctionBuilder& fb, const ValueList& args) const override;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    bool _sample;
    bool _unweight;
};

}
