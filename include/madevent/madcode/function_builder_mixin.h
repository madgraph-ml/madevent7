// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

Value add(Value in1, Value in2) {
    return instruction("add", {in1, in2})[0];
}

Value sub(Value in1, Value in2) {
    return instruction("sub", {in1, in2})[0];
}

Value mul(Value in1, Value in2) {
    return instruction("mul", {in1, in2})[0];
}

Value mul_scalar(Value in1, Value in2) {
    return instruction("mul_scalar", {in1, in2})[0];
}

Value clip_min(Value x, Value min) {
    return instruction("clip_min", {x, min})[0];
}

Value sqrt(Value in) {
    return instruction("sqrt", {in})[0];
}

Value square(Value in) {
    return instruction("square", {in})[0];
}

Value uniform_phi(Value in) {
    return instruction("uniform_phi", {in})[0];
}

Value uniform_phi_inverse(Value in) {
    return instruction("uniform_phi_inverse", {in})[0];
}

Value uniform_costheta(Value in) {
    return instruction("uniform_costheta", {in})[0];
}

Value uniform_costheta_inverse(Value in) {
    return instruction("uniform_costheta_inverse", {in})[0];
}

Value rotate_zy(Value p, Value phi, Value cos_theta) {
    return instruction("rotate_zy", {p, phi, cos_theta})[0];
}

Value rotate_zy_inverse(Value p, Value phi, Value cos_theta) {
    return instruction("rotate_zy_inverse", {p, phi, cos_theta})[0];
}

Value boost(Value p1, Value p2) {
    return instruction("boost", {p1, p2})[0];
}

Value boost_inverse(Value p1, Value p2) {
    return instruction("boost_inverse", {p1, p2})[0];
}

Value boost_beam(Value p1, Value rap) {
    return instruction("boost_beam", {p1, rap})[0];
}

Value boost_beam_inverse(Value p1, Value rap) {
    return instruction("boost_beam_inverse", {p1, rap})[0];
}

Value com_momentum(Value sqrt_s) {
    return instruction("com_momentum", {sqrt_s})[0];
}

std::array<Value, 2> com_p_in(Value e_cm) {
    auto output_vector = instruction("com_p_in", {e_cm});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> com_angles(Value p) {
    auto output_vector = instruction("com_angles", {p});
    return {output_vector[0], output_vector[1]};
}

Value s(Value p) {
    return instruction("s", {p})[0];
}

Value sqrt_s(Value p) {
    return instruction("sqrt_s", {p})[0];
}

std::array<Value, 2> s_and_sqrt_s(Value p) {
    auto output_vector = instruction("s_and_sqrt_s", {p});
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

Value rapidity(Value x1, Value x2) {
    return instruction("rapidity", {x1, x2})[0];
}

std::array<Value, 2> decay_momentum(Value s, Value sqrt_s, Value m1, Value m2) {
    auto output_vector = instruction("decay_momentum", {s, sqrt_s, m1, m2});
    return {output_vector[0], output_vector[1]};
}

std::array<Value, 2> invt_min_max(Value s, Value s_in1, Value s_in2, Value m1, Value m2) {
    auto output_vector = instruction("invt_min_max", {s, s_in1, s_in2, m1, m2});
    return {output_vector[0], output_vector[1]};
}

Value invt_to_costheta(Value s, Value s_in1, Value s_in2, Value m1, Value m2, Value t) {
    return instruction("invt_to_costheta", {s, s_in1, s_in2, m1, m2, t})[0];
}

Value costheta_to_invt(Value s, Value s_in1, Value s_in2, Value m1, Value m2, Value cos_theta) {
    return instruction("costheta_to_invt", {s, s_in1, s_in2, m1, m2, cos_theta})[0];
}

Value two_particle_density_inverse(Value s, Value m1, Value m2) {
    return instruction("two_particle_density_inverse", {s, m1, m2})[0];
}

Value tinv_two_particle_density(Value det_t, Value s, Value s_in1, Value s_in2) {
    return instruction("tinv_two_particle_density", {det_t, s, s_in1, s_in2})[0];
}

Value tinv_two_particle_density_inverse(Value det_t, Value s, Value s_in1, Value s_in2) {
    return instruction("tinv_two_particle_density_inverse", {det_t, s, s_in1, s_in2})[0];
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

