// This file was automatically generated from the contents of instruction_set.yaml
// Do not modify its content directly

using SigType = SimpleInstruction::SigType;
const SimpleInstruction::SigType all {DT_FLOAT, {"..."}};
const SimpleInstruction::SigType scalar {DT_FLOAT, {}};
const SimpleInstruction::SigType four_vector {DT_FLOAT, {4}};
const SimpleInstruction::SigType four_vector_array {DT_FLOAT, {"n", 4}};
const auto mi = [](
    std::string name,
    int opcode,
    std::initializer_list<SigType> inputs,
    std::initializer_list<SigType> outputs
) { return InstructionPtr(new SimpleInstruction(name, opcode, inputs, outputs)); };

InstructionPtr instructions[] {
    mi("add", 0, {all, all}, {all}),
    mi("sub", 1, {all, all}, {all}),
    mi("mul", 2, {all, all}, {all}),
    mi("mul_scalar", 3, {all, scalar}, {all}),
    mi("clip_min", 4, {scalar, scalar}, {scalar}),
    mi("sqrt", 5, {scalar}, {scalar}),
    mi("square", 6, {scalar}, {scalar}),
    mi("uniform_phi", 7, {scalar}, {scalar}),
    mi("uniform_phi_inverse", 8, {scalar}, {scalar}),
    mi("uniform_costheta", 9, {scalar}, {scalar}),
    mi("uniform_costheta_inverse", 10, {scalar}, {scalar}),
    mi("rotate_zy", 11, {four_vector, scalar, scalar}, {four_vector}),
    mi("rotate_zy_inverse", 12, {four_vector, scalar, scalar}, {four_vector}),
    mi("boost", 13, {four_vector, four_vector}, {four_vector}),
    mi("boost_inverse", 14, {four_vector, four_vector}, {four_vector}),
    mi("boost_beam", 15, {four_vector_array, scalar}, {four_vector_array}),
    mi("boost_beam_inverse", 16, {four_vector_array, scalar}, {four_vector_array}),
    mi("com_momentum", 17, {scalar}, {four_vector}),
    mi("com_p_in", 18, {scalar}, {four_vector, four_vector}),
    mi("com_angles", 19, {four_vector}, {scalar, scalar}),
    mi("s", 20, {four_vector}, {scalar}),
    mi("sqrt_s", 21, {four_vector}, {scalar}),
    mi("s_and_sqrt_s", 22, {four_vector}, {scalar, scalar}),
    mi("r_to_x1x2", 23, {scalar, scalar, scalar}, {scalar, scalar, scalar}),
    mi("x1x2_to_r", 24, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("rapidity", 25, {scalar, scalar}, {scalar}),
    mi("decay_momentum", 26, {scalar, scalar, scalar, scalar}, {four_vector}),
    mi("invt_min_max", 27, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("invt_to_costheta", 28, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("costheta_to_invt", 29, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("two_particle_density", 30, {scalar, scalar, scalar}, {scalar}),
    mi("two_particle_density_inverse", 31, {scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density", 32, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density_inverse", 33, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("uniform_invariant", 34, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("uniform_invariant_inverse", 35, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant", 36, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant_inverse", 37, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant", 38, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_inverse", 39, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu", 40, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu_inverse", 41, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
};
