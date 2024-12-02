// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

using SigType = SimpleInstruction::SigType;
const SimpleInstruction::SigType all {DT_FLOAT, {"..."}};
const SimpleInstruction::SigType scalar {DT_FLOAT, {}};
const SimpleInstruction::SigType array {DT_FLOAT, {"n"}};
const SimpleInstruction::SigType array_plus {DT_FLOAT, {"n+"}};
const SimpleInstruction::SigType array_plus_2 {DT_FLOAT, {"n++"}};
const SimpleInstruction::SigType four_vector {DT_FLOAT, {4}};
const SimpleInstruction::SigType four_vector_array {DT_FLOAT, {"n", 4}};
const SimpleInstruction::SigType four_vector_array_plus {DT_FLOAT, {"n+", 4}};
const auto mi = [](
    std::string name,
    int opcode,
    std::initializer_list<SigType> inputs,
    std::initializer_list<SigType> outputs
) { return InstructionOwner(new SimpleInstruction(name, opcode, inputs, outputs)); };

InstructionOwner instructions[] {
    InstructionOwner(new StackInstruction(0)),
    InstructionOwner(new UnstackInstruction(1)),
    InstructionOwner(new BatchCatInstruction(2)),
    InstructionOwner(new BatchSplitInstruction(3)),
    mi("add", 4, {all, all}, {all}),
    mi("sub", 5, {all, all}, {all}),
    mi("mul", 6, {scalar, scalar}, {scalar}),
    mi("mul_scalar", 7, {all, scalar}, {all}),
    mi("clip_min", 8, {scalar, scalar}, {scalar}),
    mi("sqrt", 9, {scalar}, {scalar}),
    mi("square", 10, {scalar}, {scalar}),
    mi("pow", 11, {scalar, scalar}, {scalar}),
    mi("uniform_phi", 12, {scalar}, {scalar}),
    mi("uniform_phi_inverse", 13, {scalar}, {scalar}),
    mi("uniform_costheta", 14, {scalar}, {scalar}),
    mi("uniform_costheta_inverse", 15, {scalar}, {scalar}),
    mi("rotate_zy", 16, {four_vector, scalar, scalar}, {four_vector}),
    mi("rotate_zy_inverse", 17, {four_vector, scalar, scalar}, {four_vector}),
    mi("boost", 18, {four_vector, four_vector}, {four_vector}),
    mi("boost_inverse", 19, {four_vector, four_vector}, {four_vector}),
    mi("boost_beam", 20, {four_vector_array, scalar}, {four_vector_array}),
    mi("boost_beam_inverse", 21, {four_vector_array, scalar}, {four_vector_array}),
    mi("com_momentum", 22, {scalar}, {four_vector}),
    mi("com_p_in", 23, {scalar}, {four_vector, four_vector}),
    mi("com_angles", 24, {four_vector}, {scalar, scalar}),
    mi("s", 25, {four_vector}, {scalar}),
    mi("sqrt_s", 26, {four_vector}, {scalar}),
    mi("s_and_sqrt_s", 27, {four_vector}, {scalar, scalar}),
    mi("r_to_x1x2", 28, {scalar, scalar, scalar}, {scalar, scalar, scalar}),
    mi("x1x2_to_r", 29, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("rapidity", 30, {scalar, scalar}, {scalar}),
    mi("decay_momentum", 31, {scalar, scalar, scalar, scalar}, {four_vector, scalar}),
    mi("invt_min_max", 32, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("invt_to_costheta", 33, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("costheta_to_invt", 34, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("two_particle_density_inverse", 35, {scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density", 36, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density_inverse", 37, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("uniform_invariant", 38, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("uniform_invariant_inverse", 39, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant", 40, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant_inverse", 41, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant", 42, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_inverse", 43, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu", 44, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu_inverse", 45, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("fast_rambo_r_to_u", 46, {array}, {array, scalar}),
    mi("rambo_four_vectors_massless", 47, {array, scalar, array_plus, array_plus}, {four_vector_array_plus, four_vector_array_plus}),
    mi("rambo_four_vectors_massive", 48, {array, scalar, array_plus, array_plus, array_plus_2}, {four_vector_array_plus, four_vector_array_plus, scalar, scalar}),
};
