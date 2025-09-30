// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    op_stack(instr, locals, device);
    break;
case 1:
    op_unstack(instr, locals, device);
    break;
case 2:
    op_unstack_sizes(instr, locals, device);
    break;
case 3:
    op_pop(instr, locals, device);
    break;
case 4:
    op_batch_cat(instr, locals, device);
    break;
case 5:
    op_batch_split(instr, locals, device);
    break;
case 6:
    op_cat(instr, locals, device);
    break;
case 7:
    op_batch_size(instr, locals, device);
    break;
case 8:
    op_full(instr, locals, device);
    break;
case 9:
    op_squeeze(instr, locals, device);
    break;
case 10:
    op_unsqueeze(instr, locals, device);
    break;
case 11:
    batch_foreach<tensor_foreach_dynamic<kernel_add<CudaTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 12:
    batch_foreach<tensor_foreach_dynamic<kernel_sub<CudaTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 13:
    batch_foreach<tensor_foreach_dynamic<kernel_mul<CudaTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 14:
    batch_foreach<tensor_foreach<kernel_reduce_product<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 15:
    batch_foreach<tensor_foreach<kernel_sqrt<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 16:
    batch_foreach<tensor_foreach<kernel_square<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 17:
    batch_foreach<tensor_foreach_dynamic<kernel_min<CudaTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 18:
    batch_foreach<tensor_foreach_dynamic<kernel_max<CudaTypes>, 2, 1>, 2, 1>(instr, locals, device);
    break;
case 19:
    batch_foreach<tensor_foreach<kernel_boost_beam<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 20:
    batch_foreach<tensor_foreach<kernel_boost_beam_inverse<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 21:
    batch_foreach<tensor_foreach<kernel_com_p_in<CudaTypes>, 1, 2, 1>, 1, 2>(instr, locals, device);
    break;
case 22:
    batch_foreach<tensor_foreach<kernel_r_to_x1x2<CudaTypes>, 3, 3, 1>, 3, 3>(instr, locals, device);
    break;
case 23:
    batch_foreach<tensor_foreach<kernel_x1x2_to_r<CudaTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 24:
    batch_foreach<tensor_foreach<kernel_diff_cross_section<CudaTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 25:
    batch_foreach<tensor_foreach<kernel_two_particle_decay_com<CudaTypes>, 5, 3, 1>, 5, 3>(instr, locals, device);
    break;
case 26:
    batch_foreach<tensor_foreach<kernel_two_particle_decay<CudaTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 27:
    batch_foreach<tensor_foreach<kernel_two_particle_scattering_com<CudaTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 28:
    batch_foreach<tensor_foreach<kernel_two_particle_scattering<CudaTypes>, 6, 3, 1>, 6, 3>(instr, locals, device);
    break;
case 29:
    batch_foreach<tensor_foreach<kernel_t_inv_min_max<CudaTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 30:
    batch_foreach<tensor_foreach<kernel_invariants_from_momenta<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 31:
    batch_foreach<tensor_foreach<kernel_sde2_channel_weights<CudaTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 32:
    batch_foreach<tensor_foreach<kernel_subchannel_weights<CudaTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 33:
    batch_foreach<tensor_foreach<kernel_apply_subchannel_weights<CudaTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 34:
    batch_foreach<tensor_foreach<kernel_pt_eta_phi_x<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 35:
    batch_foreach<tensor_foreach<kernel_mirror_momenta<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 36:
    batch_foreach<tensor_foreach<kernel_uniform_invariant<CudaTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 37:
    batch_foreach<tensor_foreach<kernel_uniform_invariant_inverse<CudaTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 38:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant<CudaTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 39:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant_inverse<CudaTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 40:
    batch_foreach<tensor_foreach<kernel_stable_invariant<CudaTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 41:
    batch_foreach<tensor_foreach<kernel_stable_invariant_inverse<CudaTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 42:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu<CudaTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 43:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu_inverse<CudaTypes>, 5, 2, 1>, 5, 2>(instr, locals, device);
    break;
case 44:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless<CudaTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 45:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless_com<CudaTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 46:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive<CudaTypes>, 4, 2, 1>, 4, 2>(instr, locals, device);
    break;
case 47:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive_com<CudaTypes>, 3, 2, 1>, 3, 2>(instr, locals, device);
    break;
case 48:
    batch_foreach<tensor_foreach<kernel_cut_unphysical<CudaTypes>, 4, 1, 1>, 4, 1>(instr, locals, device);
    break;
case 49:
    batch_foreach<tensor_foreach<kernel_cut_pt<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 50:
    batch_foreach<tensor_foreach<kernel_cut_eta<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 51:
    batch_foreach<tensor_foreach<kernel_cut_dr<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 52:
    batch_foreach<tensor_foreach<kernel_cut_m_inv<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 53:
    batch_foreach<tensor_foreach<kernel_cut_sqrt_s<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 54:
    batch_foreach<tensor_foreach<kernel_scale_transverse_energy<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 55:
    batch_foreach<tensor_foreach<kernel_scale_transverse_mass<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 56:
    batch_foreach<tensor_foreach<kernel_scale_half_transverse_mass<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 57:
    batch_foreach<tensor_foreach<kernel_scale_partonic_energy<CudaTypes>, 1, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 58:
    batch_foreach<tensor_foreach<kernel_chili_forward<CudaTypes>, 5, 4, 1>, 5, 4>(instr, locals, device);
    break;
case 59:
    op_matrix_element(instr, locals, device);
    break;
case 60:
    op_matrix_element_multichannel(instr, locals, device);
    break;
case 61:
    batch_foreach<tensor_foreach<kernel_collect_channel_weights<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 62:
    batch_foreach<tensor_foreach<kernel_interpolate_pdf<CudaTypes>, 6, 1, 1>, 6, 1>(instr, locals, device);
    break;
case 63:
    batch_foreach<tensor_foreach<kernel_interpolate_alpha_s<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 64:
    op_matmul(instr, locals, device);
    break;
case 65:
    batch_foreach<tensor_foreach_dynamic<kernel_relu<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 66:
    batch_foreach<tensor_foreach_dynamic<kernel_leaky_relu<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 67:
    batch_foreach<tensor_foreach_dynamic<kernel_elu<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 68:
    batch_foreach<tensor_foreach_dynamic<kernel_gelu<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 69:
    batch_foreach<tensor_foreach_dynamic<kernel_sigmoid<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 70:
    batch_foreach<tensor_foreach_dynamic<kernel_softplus<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 71:
    op_rqs_reshape(instr, locals, device);
    break;
case 72:
    batch_foreach<tensor_foreach<kernel_rqs_find_bin<CudaTypes>, 4, 1, 2>, 4, 1>(instr, locals, device);
    break;
case 73:
    batch_foreach<tensor_foreach<kernel_rqs_forward<CudaTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 74:
    batch_foreach<tensor_foreach<kernel_rqs_inverse<CudaTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 75:
    batch_foreach<tensor_foreach_dynamic<kernel_softmax<CudaTypes>, 1, 1>, 1, 1>(instr, locals, device);
    break;
case 76:
    batch_foreach<tensor_foreach<kernel_softmax_prior<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 77:
    batch_foreach<tensor_foreach<kernel_sample_discrete<CudaTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 78:
    batch_foreach<tensor_foreach<kernel_sample_discrete_inverse<CudaTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 79:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs<CudaTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 80:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs_inverse<CudaTypes>, 2, 2, 1>, 2, 2>(instr, locals, device);
    break;
case 81:
    op_discrete_histogram(instr, locals, device);
    break;
case 82:
    batch_foreach<tensor_foreach<kernel_permute_momenta<CudaTypes>, 3, 1, 1>, 3, 1>(instr, locals, device);
    break;
case 83:
    batch_foreach<tensor_foreach<kernel_gather<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 84:
    batch_foreach<tensor_foreach<kernel_gather_int<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 85:
    batch_foreach<tensor_foreach<kernel_select<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 86:
    batch_foreach<tensor_foreach<kernel_one_hot<CudaTypes>, 2, 1, 1>, 2, 1>(instr, locals, device);
    break;
case 87:
    op_nonzero(instr, locals, device);
    break;
case 88:
    op_batch_gather(instr, locals, device);
    break;
case 89:
    op_batch_scatter(instr, locals, device);
    break;
case 90:
    op_random(instr, locals, device);
    break;
case 91:
    op_unweight(instr, locals, device);
    break;
case 92:
    batch_foreach<tensor_foreach<kernel_vegas_forward<CudaTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 93:
    batch_foreach<tensor_foreach<kernel_vegas_inverse<CudaTypes>, 2, 2, 2>, 2, 2>(instr, locals, device);
    break;
case 94:
    op_vegas_histogram(instr, locals, device);
    break;
