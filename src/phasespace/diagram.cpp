#include "madevent/phasespace/diagram.h"

using namespace madevent;


TPropagatorMapping::TPropagatorMapping(const Diagram& diagram, double nu, bool map_resonances) :
    Mapping(
        TypeList(4 * diagram.t_propagators.size() + 1, scalar),
        TypeList(diagram.t_propagators.size() + 3, four_vector),
        {}
    ),
    s_pseudo_invariants(diagram.t_propagators.size() - 1)
{
    // TODO: maybe allow to change integration order

    for (auto prop = diagram.t_propagators.rbegin(); prop != diagram.t_propagators.rend(); ++prop) {
        t_invariants.emplace_back(nu, prop->mass, map_resonances ? prop->width : 0.);
    }
}

Result TPropagatorMapping::build_forward_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const override {
    auto n_particles = t_invariants.size() + 1;
    auto t_inv_offset = n_particles - 2;
    auto m_out_offset = 3 * n_particles - 2;
    auto e_cm = inputs[mass_offset - 1];
    ValueList dets;

    // construct initial state momenta
    auto [p1, p2] = ir.com_p_in(e_cm);

    // sample s-invariants from the t-channel part of the diagram
    auto sqrt_s_max = e_cm;
    ValueList cumulated_m_out {inputs[m_out_offset]};
    for (int i = 0; i < n_particles - 2; ++i) {
        auto invariant = s_pseudo_invariants[i];
        auto r = inputs[i];
        auto sqrt_s = inputs[m_out_offset + 1 + i];
        auto sqrt_s_rev = inputs[m_out_offset + n_particles - 1 - i];

        sqrt_s_max = fb.sub(sqrt_s_max, sqrt_s_rev);
        auto s_max = fb.square(sqrt_s_max);
        auto s_min = fb.square(fb.add(cumulated_m_out[i], sqs));
        auto [s_vec, det] = invariant.build_forward_mapping(fb, {r}, {s_min, s_max});
        cumulated_m_out.push_back(fb.sqrt(s_vec));
        dets.push_back(det);
    }

    // sample t-invariants and build momenta of t-channel part of the diagram
    ValueList p_out;
    auto p2_rest = p2;
    for (int i = 0; i < n_particles - 1; ++i) {
        auto invariant = t_invariants[i];
        auto cum_m_out = cumulated_m_out[n_particles - 2 - i];
        auto mass = inputs[m_out_offset + n_particles - 1 - i];
        auto r1 = inputs[t_inv_offset + 2 * i];
        auto r2 = inputs[t_inv_offset + 2 * i + 1];

        auto [ks, det] = invariant.build_forward_mapping(
            fb, {r1, r2, cum_m_out, mass}, {p1, p2_rest}
        );
        auto [k_rest, k] = ks;
        p_out.push_back(k);
        p2_rest = fb.sub(p2_rest, k);
        dets.push_back(det);
    }
    p_out.push_back(k_rest);

    ValueList outputs {p1, p2};
    outputs.insert(outputs.end(), p_out.rbegin(), p_out.rend());
    return outputs, fb.product(det);
}

Result TPropagatorMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const override {

}
