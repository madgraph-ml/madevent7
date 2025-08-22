#include "madevent/phasespace/chili.h"

#include <ranges>

using namespace madevent;

ChiliMapping::ChiliMapping(
    std::size_t _n_particles, const std::vector<double>& _y_max, const std::vector<double>& _pt_min
) :
    Mapping(
        "ChiliMapping",
        TypeVec(4 * _n_particles - 1, batch_float),
        [&] {
            TypeVec output_types(_n_particles + 2, batch_four_vec);
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
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    ValueVec r, m_out;
    for (auto it = inputs.begin(); it != inputs.begin() + 3 * n_particles - 2; ++it) {
        r.push_back(*it);
    }
    auto e_cm = inputs.at(3 * n_particles - 2);
    for (auto it = inputs.begin() + 3 * n_particles - 1; it != inputs.end(); ++it) {
        m_out.push_back(*it);
    }
    auto [p_ext, x1, x2, det] = fb.chili_forward(
        fb.stack(r), e_cm, fb.stack(m_out), pt_min, y_max
    );
    auto outputs = fb.unstack(p_ext);
    outputs.push_back(x1);
    outputs.push_back(x2);
    return {outputs, det};
}

Mapping::Result ChiliMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
