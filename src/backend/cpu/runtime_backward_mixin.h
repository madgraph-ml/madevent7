// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 59:
    backward_op_matmul(instr, locals, local_grads);
    break;
case 60:
    backward_batch_foreach<backward_kernel_leaky_relu<CpuTypes>, backward_kernel_leaky_relu<SimdTypes>, 1, 1, 1, 0, 2>(instr, locals, local_grads, {0}, {});
    break;
case 61:
    backward_batch_foreach<backward_kernel_rqs_activation<CpuTypes>, backward_kernel_rqs_activation<SimdTypes>, 2, 3, 0, 2, 1>(instr, locals, local_grads, {}, {0,1});
    break;
case 62:
    backward_batch_foreach<backward_kernel_rqs_find_bin<CpuTypes>, backward_kernel_rqs_find_bin<SimdTypes>, 4, 1, 4, 0, 2>(instr, locals, local_grads, {0,1,2,3}, {});
    break;
case 63:
    backward_batch_foreach<backward_kernel_rqs_forward<CpuTypes>, backward_kernel_rqs_forward<SimdTypes>, 2, 2, 2, 0, 2>(instr, locals, local_grads, {0,1}, {});
    break;
case 64:
    backward_batch_foreach<backward_kernel_rqs_inverse<CpuTypes>, backward_kernel_rqs_inverse<SimdTypes>, 2, 2, 2, 0, 2>(instr, locals, local_grads, {0,1}, {});
    break;
case 65:
    backward_batch_foreach<backward_kernel_softmax<CpuTypes>, backward_kernel_softmax<SimdTypes>, 1, 1, 0, 1, 1>(instr, locals, local_grads, {}, {0});
    break;
case 66:
    backward_batch_foreach<backward_kernel_softmax_prior<CpuTypes>, backward_kernel_softmax_prior<SimdTypes>, 2, 1, 0, 1, 1>(instr, locals, local_grads, {}, {0});
    break;
