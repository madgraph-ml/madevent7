#pragma once

#include "madevent/phasespace/mapping.h"

#include <vector>

namespace madevent {

class ChiliMapping : public Mapping {
public:
    ChiliMapping(
        std::size_t _n_particles,
        const std::vector<double>& _y_max,
        const std::vector<double>& _pt_min
    );

    std::size_t random_dim() const {
        return 3 * n_particles - 2;
    }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override;

    std::size_t n_particles;
    std::vector<double> y_max;
    std::vector<double> pt_min;
};

}
