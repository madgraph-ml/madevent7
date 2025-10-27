// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

Value stack(ValueVec args) {
    return instruction("stack", args)[0];
}

ValueVec unstack(Value in) {
    return instruction("unstack", {in});
}

ValueVec unstack_sizes(Value in) {
    return instruction("unstack_sizes", {in});
}

std::array<Value, 2> pop(Value in) {
    auto output_vector = instruction("pop", {in});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> batch_cat(ValueVec args) {
    auto output_vector = instruction("batch_cat", args);
    return {output_vector[0], output_vector[1]};
}

ValueVec batch_split(Value in, Value counts) {
    return instruction("batch_split", {in, counts});
}

Value cat(ValueVec args) {
    return instruction("cat", args)[0];
}

Value batch_size(ValueVec args) {
    return instruction("batch_size", args)[0];
}

Value offset_indices(Value batch_sizes_offset, Value batch_sizes_out) {
    return instruction("offset_indices", {batch_sizes_offset, batch_sizes_out})[0];
}

Value full(ValueVec args) {
    return instruction("full", args)[0];
}

Value squeeze(Value input) {
    return instruction("squeeze", {input})[0];
}

Value unsqueeze(Value input) {
    return instruction("unsqueeze", {input})[0];
}

Value add(Value in1, Value in2) {
    return instruction("add", {in1, in2})[0];
}

Value add_int(Value in1, Value in2) {
    return instruction("add_int", {in1, in2})[0];
}

Value sub(Value in1, Value in2) {
    return instruction("sub", {in1, in2})[0];
}

Value mul(Value in1, Value in2) {
    return instruction("mul", {in1, in2})[0];
}

Value reduce_product(Value in) {
    return instruction("reduce_product", {in})[0];
}

Value sqrt(Value in) {
    return instruction("sqrt", {in})[0];
}

Value square(Value in) {
    return instruction("square", {in})[0];
}

Value min(Value in1, Value in2) {
    return instruction("min", {in1, in2})[0];
}

Value max(Value in1, Value in2) {
    return instruction("max", {in1, in2})[0];
}

Value boost_beam(Value p1, Value x1, Value x2) {
    return instruction("boost_beam", {p1, x1, x2})[0];
}

Value boost_beam_inverse(Value p1, Value x1, Value x2) {
    return instruction("boost_beam_inverse", {p1, x1, x2})[0];
}

std::array<Value, 2> com_p_in(Value e_cm) {
    auto output_vector = instruction("com_p_in", {e_cm});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 3> r_to_x1x2(Value r, Value s_hat, Value s_lab) {
    auto output_vector = instruction("r_to_x1x2", {r, s_hat, s_lab});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 2> x1x2_to_r(Value x1, Value x2, Value s_lab) {
    auto output_vector = instruction("x1x2_to_r", {x1, x2, s_lab});
    return {output_vector[0], output_vector[1]};
}

Value diff_cross_section(Value x1, Value x2, Value pdf1, Value pdf2, Value matrix_element, Value e_cm2) {
    return instruction("diff_cross_section", {x1, x2, pdf1, pdf2, matrix_element, e_cm2})[0];
}

std::array<Value, 3> two_body_decay_com(Value r_phi, Value r_cos_theta, Value m0, Value m1, Value m2) {
    auto output_vector = instruction("two_body_decay_com", {r_phi, r_cos_theta, m0, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_body_decay(Value r_phi, Value r_cos_theta, Value m0, Value m1, Value m2, Value p0) {
    auto output_vector = instruction("two_body_decay", {r_phi, r_cos_theta, m0, m1, m2, p0});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_to_two_particle_scattering_com(Value r_phi, Value pa, Value pb, Value t_abs, Value m1, Value m2) {
    auto output_vector = instruction("two_to_two_particle_scattering_com", {r_phi, pa, pb, t_abs, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_to_two_particle_scattering(Value r_phi, Value pa, Value pb, Value t_abs, Value m1, Value m2) {
    auto output_vector = instruction("two_to_two_particle_scattering", {r_phi, pa, pb, t_abs, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_to_three_particle_scattering(Value phi_choice, Value pa, Value pb, Value p3, Value s23, Value t1_abs, Value m1, Value m2) {
    auto output_vector = instruction("two_to_three_particle_scattering", {phi_choice, pa, pb, p3, s23, t1_abs, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 4> three_body_decay_com(Value r_e1, Value r_e2, Value r_phi, Value r_cos_theta, Value r_beta, Value m0, Value m1, Value m2, Value m3) {
    auto output_vector = instruction("three_body_decay_com", {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3]};
}

std::array<Value, 4> three_body_decay(Value r_e1, Value r_e2, Value r_phi, Value r_cos_theta, Value r_beta, Value m0, Value m1, Value m2, Value m3, Value p0) {
    auto output_vector = instruction("three_body_decay", {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3]};
}

std::array<Value, 2> t_inv_min_max(Value pa, Value pb, Value m1, Value m2) {
    auto output_vector = instruction("t_inv_min_max", {pa, pb, m1, m2});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> s_inv_min_max(Value pa, Value pb, Value p3, Value t1_abs, Value m1, Value m2) {
    auto output_vector = instruction("s_inv_min_max", {pa, pb, p3, t1_abs, m1, m2});
    return {output_vector[0], output_vector[1]};
}

Value invariants_from_momenta(Value p_ext, Value factors) {
    return instruction("invariants_from_momenta", {p_ext, factors})[0];
}

Value sde2_channel_weights(Value invariants, Value masses, Value widths, Value indices) {
    return instruction("sde2_channel_weights", {invariants, masses, widths, indices})[0];
}

Value subchannel_weights(Value invariants, Value masses, Value widths, Value indices, Value on_shell, Value group_sizes) {
    return instruction("subchannel_weights", {invariants, masses, widths, indices, on_shell, group_sizes})[0];
}

Value apply_subchannel_weights(Value channel_weights_in, Value subchannel_weights, Value channel_indices, Value subchannel_indices) {
    return instruction("apply_subchannel_weights", {channel_weights_in, subchannel_weights, channel_indices, subchannel_indices})[0];
}

Value pt_eta_phi_x(Value p_ext, Value x1, Value x2) {
    return instruction("pt_eta_phi_x", {p_ext, x1, x2})[0];
}

Value mirror_momenta(Value p_ext, Value mirror) {
    return instruction("mirror_momenta", {p_ext, mirror})[0];
}

std::array<Value, 2> uniform_invariant(Value r, Value s_min, Value s_max) {
    auto output_vector = instruction("uniform_invariant", {r, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> uniform_invariant_inverse(Value s, Value s_min, Value s_max) {
    auto output_vector = instruction("uniform_invariant_inverse", {s, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> breit_wigner_invariant(Value r, Value mass, Value width, Value s_min, Value s_max) {
    auto output_vector = instruction("breit_wigner_invariant", {r, mass, width, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> breit_wigner_invariant_inverse(Value s, Value mass, Value width, Value s_min, Value s_max) {
    auto output_vector = instruction("breit_wigner_invariant_inverse", {s, mass, width, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> stable_invariant(Value r, Value mass, Value s_min, Value s_max) {
    auto output_vector = instruction("stable_invariant", {r, mass, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> stable_invariant_inverse(Value s, Value mass, Value s_min, Value s_max) {
    auto output_vector = instruction("stable_invariant_inverse", {s, mass, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> stable_invariant_nu(Value r, Value mass, Value nu, Value s_min, Value s_max) {
    auto output_vector = instruction("stable_invariant_nu", {r, mass, nu, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> stable_invariant_nu_inverse(Value s, Value mass, Value nu, Value s_min, Value s_max) {
    auto output_vector = instruction("stable_invariant_nu_inverse", {s, mass, nu, s_min, s_max});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> fast_rambo_massless(Value r, Value e_cm, Value p0) {
    auto output_vector = instruction("fast_rambo_massless", {r, e_cm, p0});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> fast_rambo_massless_com(Value r, Value e_cm) {
    auto output_vector = instruction("fast_rambo_massless_com", {r, e_cm});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> fast_rambo_massive(Value r, Value e_cm, Value masses, Value p0) {
    auto output_vector = instruction("fast_rambo_massive", {r, e_cm, masses, p0});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> fast_rambo_massive_com(Value r, Value e_cm, Value masses) {
    auto output_vector = instruction("fast_rambo_massive_com", {r, e_cm, masses});
    return {output_vector[0], output_vector[1]};
}

Value cut_unphysical(Value w_in, Value p, Value x1, Value x2) {
    return instruction("cut_unphysical", {w_in, p, x1, x2})[0];
}

Value cut_pt(Value p, Value min_max) {
    return instruction("cut_pt", {p, min_max})[0];
}

Value cut_eta(Value p, Value min_max) {
    return instruction("cut_eta", {p, min_max})[0];
}

Value cut_dr(Value p, Value indices, Value min_max) {
    return instruction("cut_dr", {p, indices, min_max})[0];
}

Value cut_m_inv(Value p, Value indices, Value min_max) {
    return instruction("cut_m_inv", {p, indices, min_max})[0];
}

Value cut_sqrt_s(Value p, Value min_max) {
    return instruction("cut_sqrt_s", {p, min_max})[0];
}

Value scale_transverse_energy(Value momenta) {
    return instruction("scale_transverse_energy", {momenta})[0];
}

Value scale_transverse_mass(Value momenta) {
    return instruction("scale_transverse_mass", {momenta})[0];
}

Value scale_half_transverse_mass(Value momenta) {
    return instruction("scale_half_transverse_mass", {momenta})[0];
}

Value scale_partonic_energy(Value momenta) {
    return instruction("scale_partonic_energy", {momenta})[0];
}

std::array<Value, 4> chili_forward(Value r, Value e_cm, Value m_out, Value pt_min, Value y_max) {
    auto output_vector = instruction("chili_forward", {r, e_cm, m_out, pt_min, y_max});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3]};
}

Value matrix_element(Value momenta, Value flavor, Value index) {
    return instruction("matrix_element", {momenta, flavor, index})[0];
}

std::array<Value, 5> matrix_element_multichannel(Value momenta, Value alpha_s, Value random, Value flavor, Value index, Value diagram_count) {
    auto output_vector = instruction("matrix_element_multichannel", {momenta, alpha_s, random, flavor, index, diagram_count});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3], output_vector[4]};
}

Value collect_channel_weights(Value amp2, Value channel_indices, Value channel_count) {
    return instruction("collect_channel_weights", {amp2, channel_indices, channel_count})[0];
}

Value interpolate_pdf(Value x, Value q2, Value pid_indices, Value grid_logx, Value grid_logq2, Value grid_coeffs) {
    return instruction("interpolate_pdf", {x, q2, pid_indices, grid_logx, grid_logq2, grid_coeffs})[0];
}

Value interpolate_alpha_s(Value q2, Value grid_logq2, Value grid_coeffs) {
    return instruction("interpolate_alpha_s", {q2, grid_logq2, grid_coeffs})[0];
}

Value matmul(Value x, Value weight, Value bias) {
    return instruction("matmul", {x, weight, bias})[0];
}

Value relu(Value in) {
    return instruction("relu", {in})[0];
}

Value leaky_relu(Value in) {
    return instruction("leaky_relu", {in})[0];
}

Value elu(Value in) {
    return instruction("elu", {in})[0];
}

Value gelu(Value in) {
    return instruction("gelu", {in})[0];
}

Value sigmoid(Value in) {
    return instruction("sigmoid", {in})[0];
}

Value softplus(Value in) {
    return instruction("softplus", {in})[0];
}

std::array<Value, 3> rqs_reshape(Value input, Value bin_count) {
    auto output_vector = instruction("rqs_reshape", {input, bin_count});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

Value rqs_find_bin(Value input, Value in_sizes, Value out_sizes, Value derivatives) {
    return instruction("rqs_find_bin", {input, in_sizes, out_sizes, derivatives})[0];
}

std::array<Value, 2> rqs_forward(Value input, Value condition) {
    auto output_vector = instruction("rqs_forward", {input, condition});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> rqs_inverse(Value input, Value condition) {
    auto output_vector = instruction("rqs_inverse", {input, condition});
    return {output_vector[0], output_vector[1]};
}

Value softmax(Value input) {
    return instruction("softmax", {input})[0];
}

Value softmax_prior(Value input, Value prior) {
    return instruction("softmax_prior", {input, prior})[0];
}

std::array<Value, 2> sample_discrete(Value r, Value option_count) {
    auto output_vector = instruction("sample_discrete", {r, option_count});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> sample_discrete_inverse(Value index, Value option_count) {
    auto output_vector = instruction("sample_discrete_inverse", {index, option_count});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> sample_discrete_probs(Value r, Value probs) {
    auto output_vector = instruction("sample_discrete_probs", {r, probs});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> sample_discrete_probs_inverse(Value index, Value probs) {
    auto output_vector = instruction("sample_discrete_probs_inverse", {index, probs});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> discrete_histogram(Value input, Value weights, Value option_count) {
    auto output_vector = instruction("discrete_histogram", {input, weights, option_count});
    return {output_vector[0], output_vector[1]};
}

Value permute_momenta(Value momenta, Value permutations, Value index) {
    return instruction("permute_momenta", {momenta, permutations, index})[0];
}

Value gather(Value index, Value choices) {
    return instruction("gather", {index, choices})[0];
}

Value gather_int(Value index, Value choices) {
    return instruction("gather_int", {index, choices})[0];
}

Value select(Value input, Value indices) {
    return instruction("select", {input, indices})[0];
}

Value one_hot(Value index, Value option_count) {
    return instruction("one_hot", {index, option_count})[0];
}

Value nonzero(Value input) {
    return instruction("nonzero", {input})[0];
}

Value batch_gather(Value indices, Value values) {
    return instruction("batch_gather", {indices, values})[0];
}

Value batch_scatter(Value indices, Value target, Value source) {
    return instruction("batch_scatter", {indices, target, source})[0];
}

Value random(Value batch_size, Value count) {
    return instruction("random", {batch_size, count})[0];
}

std::array<Value, 2> unweight(Value weights, Value max_weight) {
    auto output_vector = instruction("unweight", {weights, max_weight});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> vegas_forward(Value input, Value grid) {
    auto output_vector = instruction("vegas_forward", {input, grid});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> vegas_inverse(Value input, Value grid) {
    auto output_vector = instruction("vegas_inverse", {input, grid});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> vegas_histogram(Value input, Value weights, Value bin_count) {
    auto output_vector = instruction("vegas_histogram", {input, weights, bin_count});
    return {output_vector[0], output_vector[1]};
}

