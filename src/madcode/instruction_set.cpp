#include "madevent/madcode/instruction_set.h"

#include <vector>

using namespace madevent;

const std::unordered_map<std::string, InstructionPtr> madevent::build_instruction_set() {
    using SigType = SimpleInstruction::SigType;
    const SigType all {DT_FLOAT, {"..."}};
    const SigType scalar {DT_FLOAT, {}};
    const SigType four_vector {DT_FLOAT, {4}};
    const SigType four_vector_array {DT_FLOAT, {"n", 4}};
    const auto mi = [](
        std::string name,
        std::initializer_list<SigType> inputs,
        std::initializer_list<SigType> outputs
    ) { return InstructionPtr(new SimpleInstruction(name, inputs, outputs)); };

    InstructionPtr instructions[] {
        // Basic instructions
        InstructionPtr(new PrintInstruction()),
        //StackInstruction(),
        //BatchCatInstruction(),
        //BatchSplitInstruction(),

        // Math
        mi("add", {all, all}, {all}),
        mi("sub", {all, all}, {all}),
        mi("mul", {scalar, scalar}, {scalar}),
        mi("mul_scalar", {all, scalar}, {all}),
        mi("clip_min", {scalar, scalar}, {scalar}),
        mi("sqrt", {scalar}, {scalar}),
        mi("square", {scalar}, {scalar}),
        mi("uniform", {scalar, scalar, scalar}, {scalar}),
        mi("uniform_inverse", {scalar, scalar, scalar}, {scalar}),

        // Kinematics
        mi("rotate_zy", {four_vector, scalar, scalar}, {four_vector}),
        mi("rotate_zy_inverse", {four_vector, scalar, scalar}, {four_vector}),
        mi("boost", {four_vector, four_vector}, {four_vector}),
        mi("boost_inverse", {four_vector, four_vector}, {four_vector}),
        mi("boost_beam", {four_vector_array, scalar}, {four_vector_array}),
        mi("boost_beam_inverse", {four_vector_array, scalar}, {four_vector_array}),
        mi("com_momentum", {scalar}, {four_vector}),
        mi("com_p_in", {scalar}, {four_vector, four_vector}),
        mi("com_angles", {four_vector}, {scalar, scalar}),
        mi("s", {four_vector}, {scalar}),
        mi("sqrt_s", {four_vector}, {scalar}),
        mi("s_and_sqrt_s", {four_vector}, {scalar, scalar}),
        mi("r_to_x1x2", {scalar, scalar, scalar}, {scalar, scalar, scalar}),
        mi("x1x2_to_r", {scalar, scalar, scalar}, {scalar, scalar}),
        mi("rapidity", {scalar, scalar}, {scalar}),

        // Two-body decays
        mi("decay_momentum", {scalar, scalar, scalar, scalar}, {four_vector}),
        mi("invt_min_max", {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("invt_to_costheta", {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
        mi("costheta_to_invt", {scalar, scalar, scalar, scalar, scalar, scalar}, {scalar}),
        mi("two_particle_density", {scalar, scalar, scalar}, {scalar}),
        mi("two_particle_density_inverse", {scalar, scalar, scalar}, {scalar}),
        mi("tinv_two_particle_density", {scalar, scalar, scalar, scalar}, {scalar}),
        mi("tinv_two_particle_density_inverse", {scalar, scalar, scalar, scalar}, {scalar}),

        // Invariants
        mi("uniform_invariant", {scalar, scalar, scalar}, {scalar, scalar}),
        mi("uniform_invariant_inverse", {scalar, scalar, scalar}, {scalar, scalar}),
        mi("breit_wigner_invariant", {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("breit_wigner_invariant_inverse", {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("stable_invariant", {scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("stable_invariant_inverse", {scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("stable_invariant_nu", {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
        mi("stable_invariant_nu_inverse", {scalar, scalar, scalar, scalar, scalar}, {scalar, scalar}),
    };

    std::unordered_map<std::string, InstructionPtr> instruction_set;
    for (auto& instruction : instructions) {
        instruction_set[instruction->name] = std::move(instruction);
    }
    return instruction_set;
}
