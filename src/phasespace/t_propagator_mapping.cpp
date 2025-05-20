#include "madevent/phasespace/t_propagator_mapping.h"

#include "madevent/util.h"

using namespace madevent;

TPropagatorMapping::TPropagatorMapping(
    const std::vector<std::size_t>& integration_order,
    double nu
) :
    Mapping(
        TypeVec(4 * integration_order.size() + 1, batch_float),
        TypeVec(integration_order.size() + 3, batch_four_vec),
        {}
    ),
    _integration_order(integration_order),
    _com_scattering(true, nu),
    _lab_scattering(false, nu)
{
    std::size_t next_index_low = 0;
    std::size_t next_index_high = integration_order.size() - 1;
    for (std::size_t index : integration_order) {
        if (index == next_index_high) {
            _sample_sides.push_back(true);
            --next_index_high;
        } else if (index == next_index_low) {
            _sample_sides.push_back(false);
            ++next_index_low;
        } else {
            throw std::invalid_argument("Invalid integration order");
        }
    }
}

Mapping::Result TPropagatorMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    ValueVec random(inputs.begin(), inputs.begin() + random_dim());
    Value e_cm = inputs.at(random_dim());
    ValueVec m_out(inputs.begin() + random_dim() + 1, inputs.end());
    auto r = random.begin();
    auto next_random = [&]() { return *(r++); };
    ValueVec dets;

    ValueVec mass_sum_invariants;
    if (_integration_order.size() > 1) {
        // compute sums of outgoing masses, starting from those sampled last
        std::size_t last_index = _integration_order.back();
        ValueVec min_masses{fb.add(m_out.at(last_index), m_out.at(last_index + 1))};
        ValueVec max_masses_subtract;
        for (std::size_t i = _integration_order.size() - 2; i > 0; --i) {
            Value next_mass = m_out.at(_integration_order.at(i) + _sample_sides.at(i));
            min_masses.push_back(fb.add(min_masses.back(), next_mass));
            max_masses_subtract.push_back(next_mass);
        }
        max_masses_subtract.push_back(
            m_out.at(_integration_order.at(0) + _sample_sides.at(0))
        );

        // sample intermediate invariant masses
        auto max_mass = e_cm;
        for (auto [min_mass, max_mass_subtract] : zip(
            std::views::reverse(min_masses), std::views::reverse(max_masses_subtract)
        )) {
            auto s_min = fb.square(min_mass);
            auto s_max = fb.square(fb.sub(max_mass, max_mass_subtract));
            auto [s_vec, det] = _uniform_invariant.build_forward(
                fb, {next_random()}, {s_min, s_max}
            );
            auto mass = fb.sqrt(s_vec.at(0));
            mass_sum_invariants.push_back(mass);
            dets.push_back(det);
            max_mass = mass;
        }
    }
    mass_sum_invariants.push_back(m_out.at(_integration_order.back()));

    // construct initial state momenta
    auto [p1, p2] = fb.com_p_in(e_cm);
    ValueVec p_ext(_integration_order.size() + 3);
    p_ext.at(0) = p1;
    p_ext.at(1) = p2;
    auto p1_rest = p1, p2_rest = p2;

    // sample t-invariants and build momenta of t-channel part of the diagram
    Value k_rest;
    bool first = true;
    for (auto [index, side, mass_sum] : zip(
        _integration_order, _sample_sides, mass_sum_invariants
    )) {
        auto& scattering = first ? _com_scattering : _lab_scattering;
        first = false;
        std::size_t sampled_index = index + side;
        auto mass = m_out.at(sampled_index);
        auto [ks, det] = scattering.build_forward(
            fb,
            {next_random(), next_random(), mass_sum, mass},
            {side ? p1_rest : p2_rest, side ? p2_rest : p1_rest}
        );
        k_rest = ks.at(0);
        auto k = ks.at(1);
        p_ext.at(sampled_index + 2) = k;
        dets.push_back(det);
        if (side) {
            p2_rest = fb.sub(p2_rest, k);
        } else {
            p1_rest = fb.sub(p1_rest, k);
        }
    }
    p_ext.at(_integration_order.back() + 2) = k_rest;
    auto det = dets.size() == 1 ? dets.at(0) : fb.product(fb.stack(dets));

    return {p_ext, det};
}

Mapping::Result TPropagatorMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
