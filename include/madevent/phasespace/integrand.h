#pragma once

#include "madevent/phasespace/phasespace.h"
#include "madevent/phasespace/vegas.h"
#include "madevent/util.h"

namespace madevent {

class DifferentialCrossSection : public FunctionGenerator {
public:
    DifferentialCrossSection(
        const std::vector<std::vector<int64_t>>& pid_options,
        std::size_t matrix_element_index,
        double e_cm2,
        double q2,
        std::size_t channel_count = 1,
        const std::vector<int64_t>& amp2_remap = {}
    );
    const std::vector<std::vector<int64_t>>& pid_options() const { return _pid_options; }
    std::size_t channel_count() const { return _channel_count; }
private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<std::vector<int64_t>> _pid_options;
    int64_t _matrix_element_index;
    double _e_cm2;
    double _q2;
    int64_t _channel_count;
    std::vector<int64_t> _amp2_remap;
};

class Unweighter : public FunctionGenerator {
public:
    Unweighter(const TypeVec& types, std::size_t particle_count);
private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
};

class Integrand : public FunctionGenerator {
public:
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping>;
    inline static const int sample = 1;
    inline static const int unweight = 2;
    inline static const int return_momenta = 4;
    inline static const int return_x1_x2 = 8;
    inline static const int return_random = 16;

    Integrand(
        const PhaseSpaceMapping& mapping,
        const DifferentialCrossSection& diff_xs,
        const AdaptiveMapping& adaptive_map,
        int flags = 0,
        const std::vector<std::size_t>& channel_indices = {{}}
    );
    std::size_t particle_count() const { return _mapping.particle_count(); }
    int flags() const { return _flags; }
    std::optional<std::string> vegas_grid_name() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->grid_name();
        } else {
            return std::nullopt;
        }
    }

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    AdaptiveMapping _adaptive_map;
    int _flags;
    std::vector<int64_t> _channel_indices;
    int64_t _random_dim;
};

}
