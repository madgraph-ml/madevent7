#include "madevent/phasespace/t_propagator_mapping.h"

using namespace madevent;

TPropagatorMapping::TPropagatorMapping(
    const std::vector<Propagator>& propagators, double nu, bool map_resonances
) :
    Mapping(
        TypeVec(4 * propagators.size() + 1, batch_float),
        TypeVec(propagators.size() + 3, batch_four_vec),
        {}
    ),
    s_pseudo_invariants(propagators.size() - 1)
{
    // TODO: maybe allow to change integration order

    bool com = true;
    for (auto prop = propagators.rbegin(); prop != propagators.rend(); ++prop) {
        t_invariants.emplace_back(com, nu); //, prop->mass, map_resonances ? prop->width : 0.);
        com = false;
    }
}

Mapping::Result TPropagatorMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    // TODO: document inputs somewhere
    // nti = len(t_invariants)
    // N = 4 * nti + 1
    // nti - 1     0
    // 2 * nti     nti - 1
    // 1           3 * nti - 1
    // nti + 1     3 * nti
    auto n_invariants = t_invariants.size();
    auto n_particles = n_invariants + 1;
    auto t_inv_offset = n_invariants - 1;
    auto m_out_offset = 3 * n_invariants;
    auto e_cm = inputs[m_out_offset - 1];
    ValueVec dets;

    // construct initial state momenta
    auto [p1, p2] = fb.com_p_in(e_cm);

    // sample s-invariants from the t-channel part of the diagram
    auto sqs_max = e_cm;
    ValueVec sqrt_s_max;
    for (int i = n_particles - 1; i > 1; --i) {
        auto sqrt_s = inputs[m_out_offset + i];
        sqs_max = fb.sub(sqs_max, sqrt_s);
        sqrt_s_max.push_back(sqs_max);
    }
    ValueVec cumulated_m_out {inputs[m_out_offset]};
    for (int i = 0; i < n_particles - 2; ++i) {
        auto invariant = s_pseudo_invariants[i];
        auto r = inputs[i];
        auto sqrt_s = inputs[m_out_offset + 1 + i];
        auto sqrt_s_rev = sqrt_s_max[n_particles - 3 - i];

        auto s_max = fb.square(sqrt_s_rev);
        auto s_min = fb.square(fb.add(cumulated_m_out[i], sqrt_s));
        auto [s_vec, det] = invariant.build_forward(fb, {r}, {s_min, s_max});
        cumulated_m_out.push_back(fb.sqrt(s_vec[0]));
        dets.push_back(det);
    }

    // sample t-invariants and build momenta of t-channel part of the diagram
    ValueVec p_out;
    auto p2_rest = p2;
    Value k_rest;
    for (int i = 0; i < n_particles - 1; ++i) {
        auto invariant = t_invariants[i];
        auto cum_m_out = cumulated_m_out[n_particles - 2 - i];
        auto mass = inputs[m_out_offset + n_particles - 1 - i];
        auto r1 = inputs[t_inv_offset + 2 * i];
        auto r2 = inputs[t_inv_offset + 2 * i + 1];

        auto [ks, det] = invariant.build_forward(
            fb, {r1, r2, cum_m_out, mass}, {p1, p2_rest}
        );
        k_rest = ks[0];
        auto k = ks[1];
        p_out.push_back(k);
        p2_rest = fb.sub(p2_rest, k);
        dets.push_back(det);
    }
    p_out.push_back(k_rest);

    ValueVec outputs {p1, p2};
    outputs.insert(outputs.end(), p_out.rbegin(), p_out.rend());
    return {outputs, fb.product(fb.stack(dets))};
}

Mapping::Result TPropagatorMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
    /*auto p_in1 = inputs[0];
    auto p_in2 = inputs[1];
    ValueVec outputs;
    ValueVec masses;
    auto e_cm = fb.sqrt_s(fb.add(p_in1, p_in2));

    auto n_invariants = t_invariants.size();
    auto n_particles = n_invariants + 1;

    for (int i = 2; i < n_particles + 2; ++i) {
        masses.push_back(fb.sqrt_s(inputs[i]));
    }

    // sample s-invariants from the t-channel part of the diagram
    auto sqrt_s_max = e_cm;
    ValueVec cumulated_m_out {inputs[m_out_offset]};
    for (int i = 0; i < n_particles - 2; ++i) {
        auto invariant = s_pseudo_invariants[i];
        auto r = inputs[i];
        auto sqrt_s = inputs[m_out_offset + 1 + i];
        auto sqrt_s_rev = inputs[m_out_offset + n_invariants - i];

        sqrt_s_max = fb.sub(sqrt_s_max, sqrt_s_rev);
        auto s_max = fb.square(sqrt_s_max);
        auto s_min = fb.square(fb.add(cumulated_m_out[i], sqrt_s));
        auto [s_vec, det] = invariant.build_forward(fb, {r}, {s_min, s_max});
        cumulated_m_out.push_back(fb.sqrt(s_vec[0]));
        dets.push_back(det);
    }

    // sample t-invariants and build momenta of t-channel part of the diagram
    ValueVec p_out;
    auto p2_rest = p2;
    Value k_rest;
    for (int i = 0; i < n_particles - 1; ++i) {
        auto invariant = t_invariants[i];
        auto cum_m_out = cumulated_m_out[n_particles - 2 - i];
        auto mass = inputs[m_out_offset + n_particles - 1 - i];
        auto r1 = inputs[t_inv_offset + 2 * i];
        auto r2 = inputs[t_inv_offset + 2 * i + 1];

        auto [ks, det] = invariant.build_forward(
            fb, {r1, r2, cum_m_out, mass}, {p1, p2_rest}
        );
        k_rest = ks[0];
        auto k = ks[1];
        p_out.push_back(k);
        p2_rest = fb.sub(p2_rest, k);
        dets.push_back(det);
    }
    p_out.push_back(k_rest);

    ValueVec outputs {p1, p2};
    outputs.insert(outputs.end(), p_out.rbegin(), p_out.rend());
    return {outputs, fb.product(dets)};

    outputs.push_back(e_cm);
    outputs.insert(outputs.end(), masses.begin(), masses.end());*/
}
