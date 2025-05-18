// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    backward_op_stack(instr, locals, local_grads, device);
    break;
case 1:
    backward_op_unstack(instr, locals, local_grads, device);
    break;
case 2:
    backward_op_pop(instr, locals, local_grads, device);
    break;
case 3:
    backward_op_batch_cat(instr, locals, local_grads, device);
    break;
case 4:
    backward_op_batch_split(instr, locals, local_grads, device);
    break;
case 5:
    backward_op_cat(instr, locals, local_grads, device);
    break;
case 9:
    backward_batch_foreach<tensor_foreach<backward_kernel_product<CudaTypes>, 2, 1, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 47:
    backward_op_matmul(instr, locals, local_grads, device);
    break;
case 48:
    backward_batch_foreach<tensor_foreach<backward_kernel_leaky_relu<CudaTypes>, 2, 1, 2>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 49:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_activation<CudaTypes>, 5, 2, 1>, 2, 3, 0, 2>(instr, locals, local_grads, {}, {0,1}, device);
    break;
case 50:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_find_bin<CudaTypes>, 5, 4, 2>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 51:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_forward<CudaTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 52:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_inverse<CudaTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 53:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax<CudaTypes>, 2, 1, 1>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 54:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax_prior<CudaTypes>, 2, 2, 1>, 2, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 60:
    backward_batch_foreach<tensor_foreach<backward_kernel_select<CudaTypes>, 2, 2, 1>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
