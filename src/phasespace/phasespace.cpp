#include "madevent/phasespace/phasespace.h"

#include <cmath>
#include <numeric>
#include <algorithm>

#include "madevent/constants.h"

using namespace madevent;

PhaseSpaceMapping::PhaseSpaceMapping(
    const Topology& topology, double _s_lab, double _s_hat_min, bool _leptonic,
    double s_min_epsilon, double nu
) :
    Mapping(
        //TODO: replace with scalar array
        TypeList(3 * topology.outgoing_masses.size() - (_leptonic ? 4 : 2), scalar),
        {four_vector_array(topology.outgoing_masses.size() + 2), scalar, scalar},
        {}
    ),
    pi_factors(std::pow(2 * PI, 4 - 3 * topology.outgoing_masses.size())),
    s_lab(_s_lab),
    s_hat_min(_s_hat_min),
    leptonic(_leptonic),
    has_t_channel(topology.t_propagators.size() != 0),
    sqrt_s_epsilon(std::sqrt(s_min_epsilon)),
    outgoing_masses(topology.outgoing_masses),
    permutation(topology.permutation),
    inverse_permutation(topology.inverse_permutation)
{
    // Initialize s invariants and decay mappings
    std::vector<double> sqrt_s_min(topology.outgoing_masses);
    for (auto& layer : topology.s_decays) {
        if (!has_t_channel && &layer == &topology.s_decays.back()) {
            auto& decay_mappings = s_decays.emplace_back().emplace_back();
            decay_mappings.count = layer[0].child_count;
            decay_mappings.decay.emplace(true);
            break;
        }
        auto sqs_iter = sqrt_s_min.begin();
        std::vector<double> sqrt_s_min_new;
        auto& layer_decays = s_decays.emplace_back();
        for (auto& decay : layer) {
            auto& decay_mappings = layer_decays.emplace_back();
            decay_mappings.count = decay.child_count;
            double sqs_min_sum = 0;
            for (int i = 0; i < decay.child_count; ++i, ++sqs_iter) sqs_min_sum += *sqs_iter;
            double sqs_min = std::max(sqrt_s_epsilon, sqs_min_sum);
            sqrt_s_min_new.push_back(sqs_min);
            if (decay.child_count == 1) continue;
            double mass = decay.propagator.mass < sqs_min ? 0. : decay.propagator.mass;
            decay_mappings.invariant.emplace(nu, mass, decay.propagator.width);
            decay_mappings.decay.emplace(false);
        }
        sqrt_s_min = sqrt_s_min_new;
    }

    // Initialize luminosity and t-channel mapping
    double sqs_min_sum = std::accumulate(sqrt_s_min.begin(), sqrt_s_min.end(), 0);
    double s_hat_min = std::max(sqs_min_sum * sqs_min_sum, s_hat_min);
    if (has_t_channel) {
        if (!leptonic) {
            luminosity = Luminosity(s_lab, s_hat_min);
        }
        t_mapping = TPropagatorMapping(topology.t_propagators);
    } else if (!leptonic) {
        auto& s_line = topology.s_decays.back()[0].propagator;
        if (s_line.mass >= std::sqrt(s_hat_min)) {
            luminosity = Luminosity(s_lab, s_hat_min, s_lab, 0., s_line.mass, s_line.width);
        } else {
            luminosity = Luminosity(s_lab, s_hat_min);
        }
    }

}

Mapping::Result PhaseSpaceMapping::build_forward_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    auto r = inputs.begin();
    ValueList dets{pi_factors};
    Value x1, x2, s_hat;
    if (luminosity) {
        auto [x12s, det_lumi] = luminosity->build_forward(fb, {*(r++), *(r++)}, {});
        dets.push_back(det_lumi);
        x1 = x12s[0];
        x2 = x12s[1];
        s_hat = x12s[2];
    } else {
        x1 = 1.0;
        x2 = 1.0;
        s_hat = s_lab;
    }
    auto sqrt_s_hat = fb.sqrt(s_hat);

    // sample s-invariants from decays, starting from the final state particles
    ValueList sqrt_s;
    std::transform(
        inverse_permutation.begin(), inverse_permutation.end(), std::back_inserter(sqrt_s),
        [this](auto index) { return outgoing_masses[index]; }
    );
    struct DecayData {
        const DecayMappings& mappings;
        ValueList masses;
        std::optional<Value> s;
        std::optional<Value> sqrt_s;
    };
    std::vector<std::vector<DecayData>> decay_data;
    for (auto& layer_decays : s_decays) {
        ValueList sqrt_s_min;
        auto sqrt_s_iter = sqrt_s.begin();
        auto& layer_data = decay_data.emplace_back();
        bool skip_layer = false;
        for (auto& decay : layer_decays) {
            auto sqrt_s_iter_next = sqrt_s_iter + decay.count;
            DecayData layer_data_item{decay, {sqrt_s_iter, sqrt_s_iter_next}};
            if (layer_decays.size() == 1 && !decay.invariant) {
                layer_data_item.s = s_hat;
                layer_data_item.sqrt_s = sqrt_s_hat;
                skip_layer = true;
            }
            layer_data.push_back(layer_data_item);
            sqrt_s_min.push_back(fb.clip_min(
                fb.sum(layer_data_item.masses), decay.count > 1 ? sqrt_s_epsilon : 0.0
            ));
            sqrt_s_iter = sqrt_s_iter_next;
        }
        if (skip_layer) continue;

        ValueList sqs_min_sums;
        std::partial_sum(
            sqrt_s_min.rbegin(), sqrt_s_min.rend() - 1, std::back_inserter(sqs_min_sums),
            [&fb](Value a, Value b) { return fb.add(a, b); }
        );

        auto sqs_sum = sqrt_s_hat;
        sqrt_s.clear();
        auto sqs_min_iter = sqrt_s_min.begin();
        auto sqs_min_sums_iter = sqs_min_sums.rbegin();
        for (auto& data : layer_data) {
            auto sqs_min_item = *(sqs_min_iter++);
            auto sqs_min_sum_item = *(sqs_min_sums_iter++);
            if (data.mappings.count == 1) {
                sqrt_s.push_back(sqs_min_item);
                continue;
            }
            auto s_min = fb.square(sqs_min_item);
            auto s_max =
                &data == &layer_data.back() ?
                fb.square(sqs_sum) :
                fb.square(fb.sub(sqs_sum, sqs_min_sum_item));
            auto [s_vec, det] = data.mappings.invariant->build_forward(
                fb, {*(r++)}, {s_min, s_max}
            );
            auto sqs = fb.sqrt(s_vec[0]);
            sqrt_s.push_back(sqs);
            data.s = s_vec[0];
            data.sqrt_s = sqs;
            dets.push_back(det);
        }
    }

    ValueList p_ext;
    ValueList p_out;
    if (has_t_channel) {
        ValueList t_args;
        std::copy_n(r, t_mapping->random_dim(), std::back_inserter(t_args));
        r += t_mapping->random_dim();
        t_args.push_back(sqrt_s_hat);
        std::copy(sqrt_s.begin(), sqrt_s.end(), std::back_inserter(t_args));
        auto [ps, det] = t_mapping->build_forward(fb, t_args, {});
        p_ext = {ps[0], ps[1]};
        std::copy(ps.begin() + 2, ps.end(), std::back_inserter(p_out));
    } else {
        auto [p1, p2] = fb.com_p_in(sqrt_s_hat);
        p_ext = {p1, p2};
    }

    // build the momenta of the decays
    for (auto data_iter = decay_data.rbegin(); data_iter != decay_data.rend(); ++data_iter) {
        auto& layer_data = *data_iter;
        auto p_out_prev = p_out;
        p_out.clear();
        auto k_in_iter = p_out_prev.begin();
        for (auto& data : layer_data) {
            if (data.mappings.count == 1) {
                p_out.push_back(*(k_in_iter++));
                continue;
            }
            auto k_in = *(k_in_iter++);
            ValueList decay_args{*(r++), *(r++)};
            if (k_in_iter != p_out_prev.end()) decay_args.push_back(*(k_in_iter++));
            decay_args.push_back(*data.s);
            decay_args.push_back(*data.sqrt_s);
            std::copy(data.masses.begin(), data.masses.end(), std::back_inserter(decay_args));
            auto [k_out, det] = data.mappings.decay->build_forward(fb, decay_args, {});
            std::copy(k_out.begin(), k_out.end(), std::back_inserter(p_out));
            dets.push_back(det);
        }
    }

    // permute and return momenta
    std::transform(
        permutation.begin(), permutation.end(), std::back_inserter(p_ext),
        [&p_out](auto index) { return p_out[index]; }
    );
    auto p_ext_stack = fb.stack(p_ext);
    auto p_ext_lab = luminosity ? fb.boost_beam(p_ext_stack, fb.rapidity(x1, x2)) : p_ext_stack;
    auto ps_weight = fb.product(dets);
    return {{p_ext_lab, x1, x2}, ps_weight};
}

Mapping::Result PhaseSpaceMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {

}
