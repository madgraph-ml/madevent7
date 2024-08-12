#pragma once

#include <unordered_map>

#include "madevent/madcode/type.h"
#include "madevent/madcode/instruction.h"

#define INSTRUCTIONS \
    X(print, -1, 0) \
    X(uniform, 3, 1) \
    X(uniform_inverse, 3, 1) \
    X(add, 2, 1) \
    X(sub, 2, 1) \
    X(mul, 2, 1) \
    X(mul_scalar, 2, 1) \
    X(clip_min, 2, 1) \
    X(sqrt, 1, 1) \
    X(square, 1, 1) \
    X(rotate_zy, 3, 1) \
    X(rotate_zy_inverse, 3, 1) \
    X(boost, 2, 1) \
    X(boost_inverse, 2, 1) \
    X(boost_beam, 2, 1) \
    X(boost_beam_inverse, 2, 1) \
    X(com_momentum, 1, 1) \
    X(com_p_in, 1, 2) \
    X(com_angles, 1, 2) \
    X(s, 1, 1) \
    X(sqrt_s, 1, 1) \
    X(s_and_sqrt_s, 1, 2) \
    X(r_to_x1x2, 3, 3) \
    X(x1x2_to_r, 3, 2) \
    X(rapidity, 2, 1) \
    X(decay_momentum, 4, 1) \
    X(invt_min_max, 5, 2) \
    X(invt_to_costheta, 6, 1) \
    X(costheta_to_invt, 6, 1) \
    X(two_particle_density, 3, 1) \
    X(two_particle_density_inverse, 3, 1) \
    X(tinv_two_particle_density, 4, 1) \
    X(tinv_two_particle_density_inverse, 4, 1) \
    X(uniform_invariant, 3, 2) \
    X(uniform_invariant_inverse, 3, 2) \
    X(breit_wigner_invariant, 5, 2) \
    X(breit_wigner_invariant_inverse, 5, 2) \
    X(stable_invariant, 4, 2) \
    X(stable_invariant_inverse, 4, 2) \
    X(stable_invariant_nu, 5, 2) \
    X(stable_invariant_nu_inverse, 5, 2)

namespace madevent {

using InstructionPtr = std::unique_ptr<Instruction>;
const std::unordered_map<std::string, InstructionPtr> build_instruction_set();
const std::unordered_map<std::string, InstructionPtr> instruction_set = build_instruction_set();

}
