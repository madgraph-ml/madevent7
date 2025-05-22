// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

Value stack(ValueVec args) {
    return instruction("stack", args)[0];
}

ValueVec unstack(Value in) {
    return instruction("unstack", {in});
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

Value add(Value in1, Value in2) {
    return instruction("add", {in1, in2})[0];
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

std::array<Value, 3> two_particle_decay_com(Value r_phi, Value r_cos_theta, Value m0, Value m1, Value m2) {
    auto output_vector = instruction("two_particle_decay_com", {r_phi, r_cos_theta, m0, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_particle_decay(Value r_phi, Value r_cos_theta, Value m0, Value m1, Value m2, Value p0) {
    auto output_vector = instruction("two_particle_decay", {r_phi, r_cos_theta, m0, m1, m2, p0});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_particle_scattering_com(Value r_phi, Value pa, Value pb, Value t, Value m1, Value m2) {
    auto output_vector = instruction("two_particle_scattering_com", {r_phi, pa, pb, t, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 3> two_particle_scattering(Value r_phi, Value pa, Value pb, Value t, Value m1, Value m2) {
    auto output_vector = instruction("two_particle_scattering", {r_phi, pa, pb, t, m1, m2});
    return {output_vector[0], output_vector[1], output_vector[2]};
}

std::array<Value, 2> t_inv_min_max(Value pa, Value pb, Value m1, Value m2) {
    auto output_vector = instruction("t_inv_min_max", {pa, pb, m1, m2});
    return {output_vector[0], output_vector[1]};
}

Value invariants_from_momenta(Value p_ext, Value factors) {
    return instruction("invariants_from_momenta", {p_ext, factors})[0];
}

Value sde2_channel_weights(Value invariants, Value masses, Value widths, Value indices) {
    return instruction("sde2_channel_weights", {invariants, masses, widths, indices})[0];
}

Value pt_eta_phi_x(Value p_ext, Value x1, Value x2) {
    return instruction("pt_eta_phi_x", {p_ext, x1, x2})[0];
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

std::array<Value, 4> chili_forward(Value r, Value e_cm, Value m_out, Value pt_min, Value y_max) {
    auto output_vector = instruction("chili_forward", {r, e_cm, m_out, pt_min, y_max});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3]};
}

Value matrix_element(Value momenta, Value flavor, Value mirror, Value index) {
    return instruction("matrix_element", {momenta, flavor, mirror, index})[0];
}

std::array<Value, 4> matrix_element_multichannel(Value momenta, Value alpha_s, Value random, Value flavor, Value mirror, Value amp2_remap, Value index, Value channel_count) {
    auto output_vector = instruction("matrix_element_multichannel", {momenta, alpha_s, random, flavor, mirror, amp2_remap, index, channel_count});
    return {output_vector[0], output_vector[1], output_vector[2], output_vector[3]};
}

Value pdf(Value x, Value q2, Value pid) {
    return instruction("pdf", {x, q2, pid})[0];
}

Value matmul(Value x, Value weight, Value bias) {
    return instruction("matmul", {x, weight, bias})[0];
}

Value leaky_relu(Value in) {
    return instruction("leaky_relu", {in})[0];
}

std::array<Value, 3> rqs_activation(Value input, Value bin_count) {
    auto output_vector = instruction("rqs_activation", {input, bin_count});
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

Value scatter(Value indices, Value target, Value source) {
    return instruction("scatter", {indices, target, source})[0];
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

