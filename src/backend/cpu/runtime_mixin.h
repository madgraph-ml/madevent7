// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    batch_foreach<kernel_add, 2, 1, true>(instr, locals);
    break;
case 1:
    batch_foreach<kernel_sub, 2, 1, true>(instr, locals);
    break;
case 2:
    batch_foreach<kernel_mul, 2, 1, false>(instr, locals);
    break;
case 3:
    batch_foreach<kernel_mul_scalar, 2, 1, true>(instr, locals);
    break;
case 4:
    batch_foreach<kernel_clip_min, 2, 1, false>(instr, locals);
    break;
case 5:
    batch_foreach<kernel_sqrt, 1, 1, false>(instr, locals);
    break;
case 6:
    batch_foreach<kernel_square, 1, 1, false>(instr, locals);
    break;
case 7:
    batch_foreach<kernel_uniform_phi, 1, 1, false>(instr, locals);
    break;
case 8:
    batch_foreach<kernel_uniform_phi_inverse, 1, 1, false>(instr, locals);
    break;
case 9:
    batch_foreach<kernel_uniform_costheta, 1, 1, false>(instr, locals);
    break;
case 10:
    batch_foreach<kernel_uniform_costheta_inverse, 1, 1, false>(instr, locals);
    break;
case 11:
    batch_foreach<kernel_rotate_zy, 3, 1, false>(instr, locals);
    break;
case 12:
    batch_foreach<kernel_rotate_zy_inverse, 3, 1, false>(instr, locals);
    break;
case 13:
    batch_foreach<kernel_boost, 2, 1, false>(instr, locals);
    break;
case 14:
    batch_foreach<kernel_boost_inverse, 2, 1, false>(instr, locals);
    break;
case 15:
    batch_foreach<kernel_boost_beam, 2, 1, false>(instr, locals);
    break;
case 16:
    batch_foreach<kernel_boost_beam_inverse, 2, 1, false>(instr, locals);
    break;
case 17:
    batch_foreach<kernel_com_momentum, 1, 1, false>(instr, locals);
    break;
case 18:
    batch_foreach<kernel_com_p_in, 1, 2, false>(instr, locals);
    break;
case 19:
    batch_foreach<kernel_com_angles, 1, 2, false>(instr, locals);
    break;
case 20:
    batch_foreach<kernel_s, 1, 1, false>(instr, locals);
    break;
case 21:
    batch_foreach<kernel_sqrt_s, 1, 1, false>(instr, locals);
    break;
case 22:
    batch_foreach<kernel_s_and_sqrt_s, 1, 2, false>(instr, locals);
    break;
case 23:
    batch_foreach<kernel_r_to_x1x2, 3, 3, false>(instr, locals);
    break;
case 24:
    batch_foreach<kernel_x1x2_to_r, 3, 2, false>(instr, locals);
    break;
case 25:
    batch_foreach<kernel_rapidity, 2, 1, false>(instr, locals);
    break;
case 26:
    batch_foreach<kernel_decay_momentum, 4, 2, false>(instr, locals);
    break;
case 27:
    batch_foreach<kernel_invt_min_max, 5, 2, false>(instr, locals);
    break;
case 28:
    batch_foreach<kernel_invt_to_costheta, 6, 1, false>(instr, locals);
    break;
case 29:
    batch_foreach<kernel_costheta_to_invt, 6, 1, false>(instr, locals);
    break;
case 30:
    batch_foreach<kernel_two_particle_density_inverse, 3, 1, false>(instr, locals);
    break;
case 31:
    batch_foreach<kernel_tinv_two_particle_density, 4, 1, false>(instr, locals);
    break;
case 32:
    batch_foreach<kernel_tinv_two_particle_density_inverse, 4, 1, false>(instr, locals);
    break;
case 33:
    batch_foreach<kernel_uniform_invariant, 3, 2, false>(instr, locals);
    break;
case 34:
    batch_foreach<kernel_uniform_invariant_inverse, 3, 2, false>(instr, locals);
    break;
case 35:
    batch_foreach<kernel_breit_wigner_invariant, 5, 2, false>(instr, locals);
    break;
case 36:
    batch_foreach<kernel_breit_wigner_invariant_inverse, 5, 2, false>(instr, locals);
    break;
case 37:
    batch_foreach<kernel_stable_invariant, 4, 2, false>(instr, locals);
    break;
case 38:
    batch_foreach<kernel_stable_invariant_inverse, 4, 2, false>(instr, locals);
    break;
case 39:
    batch_foreach<kernel_stable_invariant_nu, 5, 2, false>(instr, locals);
    break;
case 40:
    batch_foreach<kernel_stable_invariant_nu_inverse, 5, 2, false>(instr, locals);
    break;
