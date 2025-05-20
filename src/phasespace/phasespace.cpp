#include "madevent/phasespace/phasespace.h"
#include "madevent/util.h"
#include "madevent/constants.h"

using namespace madevent;

namespace {

struct DecayData {
    const Topology::Decay& decay;
    std::optional<Value> mass;
    std::optional<Value> mass2;
    std::vector<Value> min_masses;
    std::optional<Value> max_mass;
    std::vector<Value> max_mass_subtract;
    std::optional<Value> momentum;

    DecayData(const Topology::Decay& decay) : decay(decay) {}
};

void update_mass_min_max(
    std::vector<DecayData>& decay_data, std::size_t decay_index
) {
    // Update the minimum mass for the entire decay tree, based on the external
    // masses and already sampled masses.
    for (auto& data : std::views::reverse(decay_data)) {
        data.min_masses.clear();
        if (data.mass) {
            data.min_masses.push_back(data.mass.value());
        } else {
            for (std::size_t child_index : data.decay.child_indices) {
                auto& child_min_masses = decay_data.at(child_index).min_masses;
                data.min_masses.insert(
                    data.min_masses.end(), child_min_masses.begin(), child_min_masses.end()
                );
            }
        }
    }

    // Go up the decay tree until propagator with known mass m_i is found. Keep track of all
    // the other nodes branching off. The maximum mass is given by
    // m_max = M_i - sum_{other nodes j} m_{min,j}
    auto& start_decay = decay_data.at(decay_index);
    auto current_decay = &start_decay;
    while (!current_decay->mass) {
        std::size_t prev_index = current_decay->decay.index;
        current_decay = &decay_data.at(current_decay->decay.parent_index);
        for (std::size_t child_index : current_decay->decay.child_indices) {
            if (child_index == prev_index) continue;
            auto& child_min_masses = decay_data.at(child_index).min_masses;
            start_decay.max_mass_subtract.insert(
                start_decay.max_mass_subtract.end(),
                child_min_masses.begin(),
                child_min_masses.end()
            );
        }
    }
    start_decay.max_mass = current_decay->mass;
}

}

PhaseSpaceMapping::PhaseSpaceMapping(
    const Topology& topology,
    double s_lab,
    bool leptonic,
    double nu,
    TChannelMode t_channel_mode,
    const std::optional<Cuts>& cuts,
    const std::vector<std::vector<std::size_t>>& permutations
) :
    Mapping(
        {batch_float_array(3 * topology.outgoing_masses().size() - (leptonic ? 4 : 2))},
        {
            batch_four_vec_array(topology.outgoing_masses().size() + 2),
            batch_float,
            batch_float
        },
        permutations.size() == 0 ? TypeVec{} : TypeVec{batch_int}
    ),
    _topology(topology),
    _cuts(
        cuts.value_or(Cuts(std::vector<int>(topology.outgoing_masses().size(), 0), {}))
    ),
    _pi_factors(
        std::pow(2 * PI, 4 - 3 * static_cast<int>(topology.outgoing_masses().size()))
    ),
    _s_lab(s_lab),
    _leptonic(leptonic),
    _t_mapping(std::monostate{}),
    _permutations(permutations)
{
    bool has_t_channel = _topology.t_propagator_count() > 0;
    struct DecayInfo {
        double m_min, pt_min, eta_max;
        std::optional<Invariant> invariant;
    };
    std::vector<DecayInfo> decay_info(_topology.decays().size());
    for (auto [index, m_min, pt_min, eta_max] : zip(
        _topology.outgoing_indices(), _topology.outgoing_masses(), _cuts.pt_min(), _cuts.eta_max()
    )) {
        decay_info.at(index) = {m_min, pt_min, eta_max, std::nullopt};
    }
    for (auto [decay, info] : zip(
        std::views::reverse(_topology.decays()), std::views::reverse(decay_info)
    )) {
        if (decay.child_indices.size() == 0) continue;
        if (decay.index == 0 && has_t_channel) continue;

        bool is_com_decay = decay.index == 0;
        if (decay.child_indices.size() == 2) {
            _s_decays.push_back(TwoParticleDecay(is_com_decay));
        } else {
            _s_decays.push_back(FastRamboMapping(decay.child_indices.size(), false, is_com_decay));
        }

        info.m_min = 0.;
        for (std::size_t child_index : decay.child_indices) {
            info.m_min += decay_info.at(child_index).m_min;
        }
        info.pt_min = 0.;
        info.eta_max = std::numeric_limits<double>::infinity();
        if (!is_com_decay) {
            // TODO: this part must be rewritten for subchannels etc.
            double mass, width;
            if (decay.mass <= info.m_min) {
                mass = 0;
                width = 0;
            } else {
                mass = decay.mass;
                width = decay.width;
            }
            info.invariant = Invariant(nu, mass, width);
        }
    }
    for (std::size_t index : _topology.decay_integration_order()) {
        auto& invariant = decay_info.at(index).invariant;
        if (invariant) {
            _s_invariants.push_back(invariant.value());
        }
    }

    double total_mass = std::accumulate(
        _topology.outgoing_masses().begin(), _topology.outgoing_masses().end(), 0.
    );
    double sqrt_s_hat_min = _cuts.sqrt_s_min();
    double s_hat_min = std::max(total_mass * total_mass, sqrt_s_hat_min * sqrt_s_hat_min);
    if (has_t_channel) {
        if (!_leptonic && t_channel_mode != PhaseSpaceMapping::chili) {
            _luminosity = Luminosity(_s_lab, s_hat_min); //, s_lab, 0.);
        }
        if (t_channel_mode == PhaseSpaceMapping::chili) {
            // |y| <= |eta|, so we can pass y_max = eta_max
            std::vector<double> eta_max, pt_min;
            for (std::size_t index : topology.decays().at(0).child_indices) {
                auto& info = decay_info.at(index);
                eta_max.push_back(info.eta_max);
                pt_min.push_back(info.pt_min);
            }
            _t_mapping = ChiliMapping(_topology.t_propagator_count() + 1, eta_max, pt_min);
        } else if (
            t_channel_mode == PhaseSpaceMapping::propagator ||
            topology.t_propagator_count() < 2
        ) {
            _t_mapping = TPropagatorMapping(_topology.t_integration_order(), nu);
        } else if (t_channel_mode == PhaseSpaceMapping::rambo) {
            //TODO: add massless special case
            _t_mapping = FastRamboMapping(_topology.t_propagator_count() + 1, false);
        }
    } else if (!_leptonic) {
        auto& first_decay = topology.decays().at(0);
        if (first_decay.mass > std::sqrt(s_hat_min)) {
            _luminosity = Luminosity(
                _s_lab, s_hat_min, _s_lab, 0., first_decay.mass, first_decay.width
            );
        } else {
            _luminosity = Luminosity(_s_lab, s_hat_min);
        }
    }
}

PhaseSpaceMapping::PhaseSpaceMapping(
    const std::vector<double>& external_masses,
    double s_lab,
    bool leptonic,
    double nu,
    TChannelMode mode,
    const std::optional<Cuts>& cuts
) : PhaseSpaceMapping(
    Topology(
        [&] {
            if (external_masses.size() < 4) {
                throw std::invalid_argument("The number of masses must be at least 4");
            }
            std::vector<Diagram::Vertex> vertices;
            auto n_out = external_masses.size() - 2;
            vertices.push_back({
                {Diagram::incoming, 0},
                {Diagram::propagator, 0},
                {Diagram::outgoing, 0},
            });
            for (std::size_t i = 1; i < n_out - 1; ++i) {
                vertices.push_back({
                    {Diagram::propagator, i - 1},
                    {Diagram::propagator, i},
                    {Diagram::outgoing, i},
                });
            }
            vertices.push_back({
                {Diagram::incoming, 1},
                {Diagram::propagator, n_out - 2},
                {Diagram::outgoing, n_out - 1},
            });
            return Diagram(
                {external_masses.at(0), external_masses.at(1)},
                {external_masses.begin() + 2, external_masses.end()},
                std::vector<Propagator>(n_out - 1),
                vertices
            );
        }()
    ), s_lab, leptonic, nu, mode, cuts
) {}

Mapping::Result PhaseSpaceMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto random_numbers = fb.unstack(inputs.at(0));
    auto r = random_numbers.begin();
    auto next_random = [&]() { return *(r++); };

    // Luminosity sampling in hadronic case
    ValueVec dets{_pi_factors};
    Value x1, x2, s_hat;
    if (_luminosity) {
        auto [x12s, det_lumi] = _luminosity->build_forward(
            fb, {next_random(), next_random()}, {}
        );
        dets.push_back(det_lumi);
        x1 = x12s.at(0);
        x2 = x12s.at(1);
        s_hat = x12s.at(2);
    } else {
        x1 = 1.0;
        x2 = 1.0;
        s_hat = _s_lab;
    }
    auto sqrt_s_hat = fb.sqrt(s_hat);

    // initialize masses and square masses
    std::vector<DecayData> decay_data(_topology.decays().begin(), _topology.decays().end());
    for (auto [decay_index, mass] : zip(
        _topology.outgoing_indices(), _topology.outgoing_masses()
    )) {
        auto& data = decay_data.at(decay_index);
        data.mass = mass;
        data.mass2 = mass * mass;
    }
    auto& root_data = decay_data.at(0);
    root_data.mass = sqrt_s_hat;
    root_data.mass2 = s_hat;

    // sample decay s-invariants, following the integration order
    std::size_t invariant_index = 0;
    for (std::size_t decay_index : _topology.decay_integration_order()) {
        if (decay_index == 0) continue;
        auto& decay = _topology.decays().at(decay_index);
        auto& data = decay_data.at(decay_index);
        update_mass_min_max(decay_data, decay_index);
        auto s_min = fb.square(fb.sum(data.min_masses));
        auto s_max = fb.square(fb.sub(
            data.max_mass.value(), fb.sum(data.max_mass_subtract)
        ));
        auto [s_vec, det] = _s_invariants.at(invariant_index++).build_forward(
            fb, {next_random()}, {s_min, s_max}
        );
        data.mass2 = s_vec.at(0);
        data.mass = fb.sqrt(data.mass2.value());
        dets.push_back(det);
    }

    // if required, build t-channel part of phase space mapping
    ValueVec p_ext;
    std::visit(Overloaded {
        [&](auto& t_mapping) {
            ValueVec args;
            auto momentum_count = _topology.t_propagator_count() + 1;
            for (std::size_t i = 0; i < t_mapping.random_dim(); ++i) {
                args.push_back(next_random());
            }
            args.push_back(sqrt_s_hat);
            for (std::size_t index : decay_data.at(0).decay.child_indices) {
                args.push_back(decay_data.at(index).mass.value());
            }
            auto [t_result, det] = t_mapping.build_forward(fb, args, {});
            std::size_t result_index;
            using TMapping = std::decay_t<decltype(t_mapping)>;
            if constexpr (std::is_same_v<TMapping, FastRamboMapping>) {
                auto [p1, p2] = fb.com_p_in(sqrt_s_hat);
                p_ext = {p1, p2};
                result_index = 0;
            } else {
                p_ext = {t_result.at(0), t_result.at(1)};
                result_index = 2;
            }
            for (std::size_t index : decay_data.at(0).decay.child_indices) {
                decay_data.at(index).momentum = t_result.at(result_index);
                ++result_index;
            }
            dets.push_back(det);

            if constexpr (std::is_same_v<TMapping, ChiliMapping>) {
                auto out_size = t_result.size();
                x1 = t_result.at(out_size - 2);
                x2 = t_result.at(out_size - 1);
            }
        },
        [&](std::monostate) {
            auto [p1, p2] = fb.com_p_in(sqrt_s_hat);
            p_ext = {p1, p2};
        }
    }, _t_mapping);

    // go through decays and generate momenta
    std::size_t decay_map_index = _s_decays.size();
    for (auto& data : decay_data) {
        if (data.decay.child_indices.size() == 0) continue;
        if (data.decay.index == 0 &&
            !std::holds_alternative<std::monostate>(_t_mapping)) continue;
        std::visit([&](auto& decay_map) {
            ValueVec decay_args{r, r += decay_map.random_dim()};
            decay_args.push_back(data.mass.value());
            for (std::size_t child_index : data.decay.child_indices) {
                decay_args.push_back(decay_data.at(child_index).mass.value());
            }
            if (data.decay.index != 0) decay_args.push_back(data.momentum.value());
            auto [k_out, det] = decay_map.build_forward(fb, decay_args, {});
            for (auto [child_index, k] : zip(data.decay.child_indices, k_out)) {
                decay_data.at(child_index).momentum = k;
            }
            dets.push_back(det);
        }, _s_decays.at(--decay_map_index));
    }

    // collect outgoing momenta
    for (std::size_t decay_index : _topology.outgoing_indices()) {
        p_ext.push_back(decay_data.at(decay_index).momentum.value());
    }
    auto p_ext_stack = fb.stack(p_ext);

    // permute momenta if permutations are given
    if (_permutations.size() > 1) {
        std::vector<std::vector<int64_t>> permutations;
        for (auto& perm : _permutations) {
            auto& new_perm = permutations.emplace_back();
            new_perm.push_back(0); //TODO: maybe remove this
            new_perm.push_back(1);
            for (auto index : perm) {
                new_perm.push_back(index + 2);
            }
        }
        p_ext_stack = fb.permute_momenta(
            p_ext_stack, permutations, conditions.at(0)
        );
    }

    // boost into correct frame and apply cuts
    auto p_ext_lab = _luminosity ? fb.boost_beam(p_ext_stack, x1, x2) : p_ext_stack;
    auto cut_weights = _cuts.build_function(fb, sqrt_s_hat, p_ext_lab);
    dets.insert(dets.end(), cut_weights.begin(), cut_weights.end());
    Value det_product = dets.size() == 2 ?
        fb.mul(dets.at(0), dets.at(1)) : fb.product(fb.stack(dets));
    auto ps_weight = fb.cut_unphysical(det_product, p_ext_lab, x1, x2);
    return {{p_ext_lab, x1, x2}, ps_weight};
}

Mapping::Result PhaseSpaceMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}
