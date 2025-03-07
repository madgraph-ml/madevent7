#include "madevent/phasespace/chili.h"

#include <ranges>
#include <print>

using namespace madevent;

ChiliMapping::ChiliMapping(
    std::size_t _n_particles, const std::vector<double>& _y_max, const std::vector<double>& _pt_min
) :
    Mapping(
        TypeList(4 * _n_particles - 1, batch_float),
        [&] {
            TypeList output_types(_n_particles + 2, batch_four_vec);
            output_types.push_back(batch_float);
            output_types.push_back(batch_float);
            return output_types;
        }(),
        {}
    ),
    n_particles(_n_particles),
    y_max(_y_max),
    pt_min(_pt_min)
{}

Mapping::Result ChiliMapping::build_forward_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    auto r = inputs | std::views::take(3 * n_particles - 2) | std::ranges::to<ValueList>();
    auto e_cm = inputs.at(3 * n_particles - 2);
    auto m_out = inputs | std::views::drop(3 * n_particles - 1)
                        | std::views::take(n_particles)
                        | std::ranges::to<ValueList>();
    auto [p_ext, x1, x2, det] = fb.chili_forward(fb.stack(r), e_cm, fb.stack(m_out), pt_min, y_max);
    auto outputs = fb.unstack(p_ext);
    outputs.push_back(x1);
    outputs.push_back(x2);
    return {outputs, det};
}

Mapping::Result ChiliMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
