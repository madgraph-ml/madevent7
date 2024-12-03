// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    op_stack(instr, locals);
    break;
case 1:
    op_unstack(instr, locals);
    break;
case 2:
    op_batch_cat(instr, locals);
    break;
case 3:
    op_batch_split(instr, locals);
    break;
case 4:
    batch_foreach<kernel_add<CpuTypes>, kernel_add<SimdTypes>, 2, 1, true>(instr, locals);
    break;
case 5:
    batch_foreach<kernel_sub<CpuTypes>, kernel_sub<SimdTypes>, 2, 1, true>(instr, locals);
    break;
case 6:
    batch_foreach<kernel_mul<CpuTypes>, kernel_mul<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 7:
    batch_foreach<kernel_mul_scalar<CpuTypes>, kernel_mul_scalar<SimdTypes>, 2, 1, true>(instr, locals);
    break;
case 8:
    batch_foreach<kernel_clip_min<CpuTypes>, kernel_clip_min<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 9:
    batch_foreach<kernel_sqrt<CpuTypes>, kernel_sqrt<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 10:
    batch_foreach<kernel_square<CpuTypes>, kernel_square<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 11:
    batch_foreach<kernel_pow<CpuTypes>, kernel_pow<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 12:
    batch_foreach<kernel_uniform_phi<CpuTypes>, kernel_uniform_phi<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 13:
    batch_foreach<kernel_uniform_phi_inverse<CpuTypes>, kernel_uniform_phi_inverse<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 14:
    batch_foreach<kernel_uniform_costheta<CpuTypes>, kernel_uniform_costheta<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 15:
    batch_foreach<kernel_uniform_costheta_inverse<CpuTypes>, kernel_uniform_costheta_inverse<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 16:
    batch_foreach<kernel_rotate_zy<CpuTypes>, kernel_rotate_zy<SimdTypes>, 3, 1, false>(instr, locals);
    break;
case 17:
    batch_foreach<kernel_rotate_zy_inverse<CpuTypes>, kernel_rotate_zy_inverse<SimdTypes>, 3, 1, false>(instr, locals);
    break;
case 18:
    batch_foreach<kernel_boost<CpuTypes>, kernel_boost<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 19:
    batch_foreach<kernel_boost_inverse<CpuTypes>, kernel_boost_inverse<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 20:
    batch_foreach<kernel_boost_beam<CpuTypes>, kernel_boost_beam<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 21:
    batch_foreach<kernel_boost_beam_inverse<CpuTypes>, kernel_boost_beam_inverse<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 22:
    batch_foreach<kernel_com_momentum<CpuTypes>, kernel_com_momentum<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 23:
    batch_foreach<kernel_com_p_in<CpuTypes>, kernel_com_p_in<SimdTypes>, 1, 2, false>(instr, locals);
    break;
case 24:
    batch_foreach<kernel_com_angles<CpuTypes>, kernel_com_angles<SimdTypes>, 1, 2, false>(instr, locals);
    break;
case 25:
    batch_foreach<kernel_s<CpuTypes>, kernel_s<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 26:
    batch_foreach<kernel_sqrt_s<CpuTypes>, kernel_sqrt_s<SimdTypes>, 1, 1, false>(instr, locals);
    break;
case 27:
    batch_foreach<kernel_s_and_sqrt_s<CpuTypes>, kernel_s_and_sqrt_s<SimdTypes>, 1, 2, false>(instr, locals);
    break;
case 28:
    batch_foreach<kernel_r_to_x1x2<CpuTypes>, kernel_r_to_x1x2<SimdTypes>, 3, 3, false>(instr, locals);
    break;
case 29:
    batch_foreach<kernel_x1x2_to_r<CpuTypes>, kernel_x1x2_to_r<SimdTypes>, 3, 2, false>(instr, locals);
    break;
case 30:
    batch_foreach<kernel_rapidity<CpuTypes>, kernel_rapidity<SimdTypes>, 2, 1, false>(instr, locals);
    break;
case 31:
    batch_foreach<kernel_decay_momentum<CpuTypes>, kernel_decay_momentum<SimdTypes>, 4, 2, false>(instr, locals);
    break;
case 32:
    batch_foreach<kernel_invt_min_max<CpuTypes>, kernel_invt_min_max<SimdTypes>, 5, 2, false>(instr, locals);
    break;
case 33:
    batch_foreach<kernel_invt_to_costheta<CpuTypes>, kernel_invt_to_costheta<SimdTypes>, 6, 1, false>(instr, locals);
    break;
case 34:
    batch_foreach<kernel_costheta_to_invt<CpuTypes>, kernel_costheta_to_invt<SimdTypes>, 6, 1, false>(instr, locals);
    break;
case 35:
    batch_foreach<kernel_two_particle_density_inverse<CpuTypes>, kernel_two_particle_density_inverse<SimdTypes>, 3, 1, false>(instr, locals);
    break;
case 36:
    batch_foreach<kernel_tinv_two_particle_density<CpuTypes>, kernel_tinv_two_particle_density<SimdTypes>, 4, 1, false>(instr, locals);
    break;
case 37:
    batch_foreach<kernel_tinv_two_particle_density_inverse<CpuTypes>, kernel_tinv_two_particle_density_inverse<SimdTypes>, 4, 1, false>(instr, locals);
    break;
case 38:
    batch_foreach<kernel_uniform_invariant<CpuTypes>, kernel_uniform_invariant<SimdTypes>, 3, 2, false>(instr, locals);
    break;
case 39:
    batch_foreach<kernel_uniform_invariant_inverse<CpuTypes>, kernel_uniform_invariant_inverse<SimdTypes>, 3, 2, false>(instr, locals);
    break;
case 40:
    batch_foreach<kernel_breit_wigner_invariant<CpuTypes>, kernel_breit_wigner_invariant<SimdTypes>, 5, 2, false>(instr, locals);
    break;
case 41:
    batch_foreach<kernel_breit_wigner_invariant_inverse<CpuTypes>, kernel_breit_wigner_invariant_inverse<SimdTypes>, 5, 2, false>(instr, locals);
    break;
case 42:
    batch_foreach<kernel_stable_invariant<CpuTypes>, kernel_stable_invariant<SimdTypes>, 4, 2, false>(instr, locals);
    break;
case 43:
    batch_foreach<kernel_stable_invariant_inverse<CpuTypes>, kernel_stable_invariant_inverse<SimdTypes>, 4, 2, false>(instr, locals);
    break;
case 44:
    batch_foreach<kernel_stable_invariant_nu<CpuTypes>, kernel_stable_invariant_nu<SimdTypes>, 5, 2, false>(instr, locals);
    break;
case 45:
    batch_foreach<kernel_stable_invariant_nu_inverse<CpuTypes>, kernel_stable_invariant_nu_inverse<SimdTypes>, 5, 2, false>(instr, locals);
    break;
case 46:
    batch_foreach<kernel_fast_rambo_r_to_u<CpuTypes>, kernel_fast_rambo_r_to_u<SimdTypes>, 1, 2, false>(instr, locals);
    break;
case 47:
    batch_foreach<kernel_rambo_four_vectors_massless<CpuTypes>, kernel_rambo_four_vectors_massless<SimdTypes>, 4, 2, false>(instr, locals);
    break;
case 48:
    batch_foreach<kernel_rambo_four_vectors_massive<CpuTypes>, kernel_rambo_four_vectors_massive<SimdTypes>, 5, 4, false>(instr, locals);
    break;
