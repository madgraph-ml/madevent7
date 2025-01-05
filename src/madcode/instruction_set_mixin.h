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
const SimpleInstruction::SigType limits {DT_FLOAT, {2}};
const SimpleInstruction::SigType limit_array {DT_FLOAT, {"n--", 2}};
const SimpleInstruction::SigType limit_array_m {DT_FLOAT, {"m", 2}};
const SimpleInstruction::SigType index_array_2 {DT_INT, {"m", 2}};
const SimpleInstruction::SigType index_array_k {DT_INT, {"m", "k"}};
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
    mi("clip_min", 7, {scalar, scalar}, {scalar}),
    mi("sqrt", 8, {scalar}, {scalar}),
    mi("square", 9, {scalar}, {scalar}),
    mi("pow", 10, {scalar, scalar}, {scalar}),
    mi("uniform_phi", 11, {scalar}, {scalar}),
    mi("uniform_phi_inverse", 12, {scalar}, {scalar}),
    mi("uniform_costheta", 13, {scalar}, {scalar}),
    mi("uniform_costheta_inverse", 14, {scalar}, {scalar}),
    mi("rotate_zy", 15, {four_vector, scalar, scalar}, {four_vector}),
    mi("rotate_zy_inverse", 16, {four_vector, scalar, scalar}, {four_vector}),
    mi("boost", 17, {four_vector, four_vector}, {four_vector}),
    mi("boost_inverse", 18, {four_vector, four_vector}, {four_vector}),
    mi("boost_beam", 19, {four_vector_array, scalar}, {four_vector_array}),
    mi("boost_beam_inverse", 20, {four_vector_array, scalar}, {four_vector_array}),
    mi("com_momentum", 21, {scalar}, {four_vector}),
    mi("com_p_in", 22, {scalar}, {four_vector, four_vector}),
    mi("com_angles", 23, {four_vector}, {scalar, scalar}),
    mi("s", 24, {four_vector}, {scalar}),
    mi("sqrt_s", 25, {four_vector}, {scalar}),
    mi("s_and_sqrt_s", 26, {four_vector}, {scalar, scalar}),
    mi("r_to_x1x2", 27, {scalar, scalar, scalar}, {scalar, scalar, scalar}),
    mi("x1x2_to_r", 28, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("rapidity", 29, {scalar, scalar}, {scalar}),
    mi("decay_momentum", 30, {scalar, scalar, scalar, scalar}, {four_vector, scalar}),
    mi("invt_min_max", 31, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("invt_to_costheta", 32, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("costheta_to_invt", 33, {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
    mi("two_particle_density_inverse", 34, {scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density", 35, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("tinv_two_particle_density_inverse", 36, {scalar, scalar, scalar, scalar}, {scalar}),
    mi("uniform_invariant", 37, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("uniform_invariant_inverse", 38, {scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant", 39, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("breit_wigner_invariant_inverse", 40, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant", 41, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_inverse", 42, {scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu", 43, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("stable_invariant_nu_inverse", 44, {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    mi("fast_rambo_r_to_u", 45, {array}, {array, scalar}),
    mi("rambo_four_vectors_massless", 46, {array, scalar, array_plus, array_plus}, {four_vector_array_plus, four_vector_array_plus}),
    mi("rambo_four_vectors_massive", 47, {array, scalar, array_plus, array_plus, array_plus_2}, {four_vector_array_plus, four_vector_array_plus, scalar, scalar}),
    mi("cut_pt", 48, {four_vector_array, scalar, limit_array}, {scalar}),
    mi("cut_eta", 49, {four_vector_array, scalar, limit_array}, {scalar}),
    mi("cut_dr", 50, {four_vector_array, scalar, index_array_2, limit_array_m}, {scalar}),
    mi("cut_m_inv", 51, {four_vector_array, scalar, index_array_k, limit_array_m}, {scalar}),
    mi("cut_sqrt_s", 52, {four_vector_array, scalar, limits}, {scalar}),
};
