namespace {

struct DecayData {
    std::optional<Value> mass;
    std::optional<Value> mass2;
    std::vector<Value> min_masses;
    std::optional<Value> max_mass;
    std::vector<Value> max_mass_subtract;
    std::optional<Value> momentum;
};

void update_mass_min_max(std::vector<DecayData>& decay_data) {
    _topology.visit(decay_index, [&](auto& visited_decay) {
        auto& min_mass = min_masses.at(visited_decay.index);
        min_mass.clear();
        auto& mass = masses.at(visited_decay.index);
        if (mass) {
            min_mass.push_back(*mass);
        } else {
            for (auto child_index : visited_decay.child_indices) {
                auto& child_min_mass = min_masses.at(child_index);
                min_mass.insert(
                    min_mass.end(), child_min_mass.begin(), child_min_mass.end()
                );
            }
        }
    });
}

}

Mapping::Result PhaseSpaceMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto random_numbers = fb.unstack(inputs.at(0));
    auto r = random_numbers.begin();

    // Luminosity sampling in hadronic case
    ValueVec dets{_pi_factors};
    Value x1, x2, s_hat;
    if (_luminosity) {
        auto [x12s, det_lumi] = _luminosity->build_forward(fb, {*(r++), *(r++)}, {});
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

    // initialize masses, square masses and minimal masses
    std::vector<DecayData> decay_data(_topology.decays().size());
    double total_mass_f = 0.;
    for (std::size_t decay_index : _topology.outgoing_indices()) {
        auto mass = _topology.decays().at(decay_index).mass;
        auto& data = decay_data.at(decay_index);
        data.mass = mass;
        data.mass2 = mass * mass;
        total_mass_f += mass;
    }
    Value total_mass(total_mass_f);

    // sample decay s-invariants, following the integration order
    std::size_t invariant_index = 0;
    for (std::size_t decay_index : _topology.decay_integration_order()) {
        auto& decay = _topology.decays().at(decay_index);
        auto& data = decay_data.at(decay_index);
        update_mass_min_max(data);
        auto s_min = fb.square(fb.sum(data.min_masses));
        auto s_max = fb.square(fb.sub(data.max_mass, fb.sum(data.max_mass_subtract)));
        auto [s_vec, det] = _invariants.at(invariant_index++).build_forward(
            fb, {next_random()}, {s_min, s_max}
        );
        auto m = fb.sqrt(s);
        data.mass2 = s_vec.at(0);
        data.mass = fb.sqrt(data.mass2);
    }

    // if required, build t-channel part of phase space mapping
    ValueVec p_ext;
    std::visit(Overloaded {
        [&](auto& t_mapping) {
            ValueVec args;
            auto momentum_count = _topology.t_propagators().size() + 1
            for (std::size_t i = 0; i < t_mapping.random_dim(); ++i) {
                args.push_back(next_random());
            }
            args.push_back(sqrt_s_hat);
            for (std::size_t i = 0; i < momentum_count; ++i) {
                args.push_back(decay_data.at(i).mass);
            }
            auto [t_result, det] = t_mapping.build_forward(fb, args, {});
            p_ext = {t_result.at(0), t_result.at(1)};
            for (std::size_t i = 0; i < momentum_count; ++i) {
                decay_data.at(i).momentum = t_result.at(i + 2);
            }

            if constexpr (std::is_same_v<T, ChiliMapping>) {
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

    //
    for (auto& [decay, data] : zip(_topology.decays(), decay_data)) {

    }
}
