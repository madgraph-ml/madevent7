// This file was automatically generated based on instruction_set.yaml
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
    InstructionPtr(new StackInstruction(0)),
    InstructionPtr(new UnstackInstruction(1)),
    mi("add", 2, {all, all}, {all}),
    mi("sub", 3, {all, all}, {all}),
    mi("mul", 4, {scalar, scalar}, {scalar}),
    mi("mul_scalar", 5, {all, scalar}, {all}),
    mi("clip_min", 6, {scalar, scalar}, {scalar}),
    mi("sqrt", 7, {scalar}, {scalar}),
    mi("square", 8, {scalar}, {scalar}),
    mi("uniform_phi", 9, {scalar}, {scalar}),
    mi("uniform_phi_inverse", 10, {scalar}, {scalar}),
    mi("uniform_costheta", 11, {scalar}, {scalar}),
    mi("uniform_costheta_inverse", 12, {scalar}, {scalar}),
    mi("rotate_zy", 13, {four_vector, scalar, scalar}, {four_vector}),
    mi("rotate_zy_inverse", 14, {four_vector, scalar, scalar}, {four_vector}),
    mi("boost", 15, {four_vector, four_vector}, {four_vector}),
    mi("boost_inverse", 16, {four_vector, four_vector}, {four_vector}),
    mi("boost_beam", 17, {four_vector_array, scalar}, {four_vector_array}),
    mi("boost_beam_inverse", 18, {four_vector_array, scalar}, {four_vector_array}),
    mi("com_momentum", 19, {scalar}, {four_vector}),
    mi("com_p_in", 20, {scalar}, {four_vector, four_vector}),
    mi("com_angles", 21, {four_vector}, {scalar, scalar}),
    mi("s", 22, {four_vector}, {scalar}),
    mi("sqrt_s", 23, {four_vector}, {scalar}),
    mi("s_and_sqrt_s", 24, {four_vector}, {scalar, scalar}),
    mi("r_to_x1x2", 25, {scalar, scalar, scalar}, {scalar, scalar, scalar}),
    mi("x1x2_to_r", 26, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("rapidity", 27, {scalar, scalar}, {scalar}),
    mi("decay_momentum", 28, {scalar, scalar, scalar, scalar}, {four_vector, scalar}),
    mi("invt_min_max", 29, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("invt_to_costheta", 30, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("costheta_to_invt", 31, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("two_particle_density_inverse", 32, {scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density", 33, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density_inverse", 34, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("uniform_invariant", 35, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("uniform_invariant_inverse", 36, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant", 37, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant_inverse", 38, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant", 39, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_inverse", 40, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu", 41, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu_inverse", 42, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
};
