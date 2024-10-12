#include "madevent/phasespace/phasespace.h"

#include <cmath>
#include <numeric>

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
    sqrt_s_epsilon(std::sqrt(s_min_epsilon)),
    outgoing_masses(topology.outgoing_masses),
    permutation(topology.permutation),
    inverse_permutation(topology.inverse_permutation)
{
    bool has_t_channel = topology.t_propagators.size() != 0;
    // Initialize s invariants and decay mappings
    std::vector<double> sqrt_s_min(topology.outgoing_masses);
    for (auto& layer : topology.s_decays) {
        if (!has_t_channel && &layer == &topology.s_decays.back()) {
            s_decay_invariants.emplace_back();
            s_decays.emplace_back().emplace_back(true);
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
    ValueList dets;
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
    std::vector<ValueList> decay_masses;
    std::vector<std::vector<std::tuple<Value, Value>>> decay_s_sqrt_s;
    for (auto& layer_decays : s_decays) {

    }

    /*
    # sample s-invariants from decays, starting from the final state particles
    sqrt_s = [ir.constant(
        self.diagram.outgoing[self.diagram.inverse_permutation[i]].mass
    ) for i in range(len(self.diagram.outgoing))]
    decay_masses = []
    decay_s_sqrt_s = []
    for layer_counts, layer_invariants in zip(
        self.diagram.s_decay_layers, self.s_decay_invariants
    ):
        sqrt_s_min = []
        sqrt_s_index = 0
        layer_masses = []
        for decay_count in layer_counts:
            sqs_clip = self.sqrt_s_epsilon if decay_count > 1 else 0.0
            sqrt_s_min.append(
                ir.clip_min(
                    ir_sum(ir, sqrt_s[sqrt_s_index : sqrt_s_index + decay_count]),
                    sqs_clip,
                )
            )
            layer_masses.append(sqrt_s[sqrt_s_index : sqrt_s_index + decay_count])
            sqrt_s_index += decay_count
        decay_masses.append(layer_masses)

        if len(layer_invariants) == 0:
            decay_s_sqrt_s.append([(s_hat, sqrt_s_hat)])
            assert not self.has_t_channel
            continue

        sqs_min_sums = [sqrt_s_min[-1]]
        for sqs_min in sqrt_s_min[-2:0:-1]:
            sqs_min_sums.append(ir.add(sqs_min_sums[-1], sqs_min))

        sqs_sum = sqrt_s_hat
        sqrt_s = []
        layer_s_sqrt_s = []
        invariant_iter = iter(layer_invariants)
        for i, decay_count in enumerate(layer_counts):
            if decay_count == 1:
                sqrt_s.append(sqrt_s_min[i])
                layer_s_sqrt_s.append((None, None))
                continue
            s_min = ir.square(sqrt_s_min[i])
            if i == len(layer_counts) - 1:
                s_max = ir.square(sqs_sum)
            else:
                s_max = ir.square(ir.sub(sqs_sum, sqs_min_sums[-i - 1]))
            (s,), jac = next(invariant_iter).map(ir, rand(), condition=[s_min, s_max])
            sqs = ir.sqrt(s)
            sqrt_s.append(sqs)
            layer_s_sqrt_s.append((s, sqs))
            sqs_sum = ir.sub(sqs_sum, sqs)
            dets.append(jac)
        decay_s_sqrt_s.append(layer_s_sqrt_s)

    if self.has_t_channel:
        (p1, p2, *p_out), jac = self.t_mapping.map(
            ir, [*rand(self.t_random_numbers), sqrt_s_hat, *sqrt_s]
        )
        if self.t_channel_type == "chili":
            # TODO
            x1 = p_in[:, 0, 0] * 2 / sqrt_s_hat
            x2 = p_in[:, 1, 0] * 2 / sqrt_s_hat
            x1x2 = torch.stack([x1, x2], dim=1)
        dets.append(jac)
    else:
        p1, p2 = ir.com_p_in(sqrt_s_hat)
        p_out = [None]

    # build the momenta of the decays
    for layer_counts, layer_decays, layer_masses, layer_s_sqrt_s in zip(
        reversed(self.diagram.s_decay_layers),
        reversed(self.s_decays),
        reversed(decay_masses),
        reversed(decay_s_sqrt_s),
    ):
        p_out_prev = p_out
        p_out = []
        decay_iter = iter(layer_decays)
        for count, k_in, masses, (dp_s, dp_sqrt_s) in zip(
            layer_counts, p_out_prev, layer_masses, layer_s_sqrt_s
        ):
            if count == 1:
                p_out.append(k_in)
                continue
            k_in = [] if k_in is None else [k_in]
            k_out, jac = next(decay_iter).map(
                ir, [*rand(2), *k_in, dp_s, dp_sqrt_s, *masses]
            )
            p_out.extend(k_out)
            dets.append(jac)

    # we should have consumed all the random numbers
    assert rand.empty()

    # permute and return momenta
    p_out_perm = [p_out[self.diagram.permutation[i]] for i in range(len(p_out))]
    p_ext = ir.stack(p1, p2, *p_out_perm, 0)
    p_ext_lab = p_ext if self.luminosity is None else ir.boost_beam(p_ext, rap)
    ps_weight = ir.mul_const(ir_product(ir, dets), self.pi_factors)
    return (p_ext_lab, x1, x2), ps_weight*/
}

Mapping::Result PhaseSpaceMapping::build_inverse_impl(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {

}
