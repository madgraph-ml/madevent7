#include "runtime.h"

#include <tuple>
#include <array>
#include <functional>
#include <algorithm>
#include <format>
#include <random>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include "madevent/util.h"
#include "tensor.h"
#include "device.h"
#include "../kernels/kernels.h"
#include "../kernels/operations.h"

using namespace madevent;
using namespace madevent::cuda;
using namespace madevent::kernels;

namespace {

void op_matrix_element(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    //TODO
}

void op_matrix_element_multichannel(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    //TODO
}

void op_matmul(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto bias = locals[instruction.input_indices[2]].contiguous(device);
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(DataType::dt_float, {batch_size, dims_out}, device);
    output.copy_from(bias, device);
    if (batch_size == 0) return;

    cublasHandle_t handle = instruction.runtime.cublas_handle();
    check_error(cublasSetStream(handle, device.stream()));
    double alpha = 1., beta = 1.;
    cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch_size, dims_out, dims_in,
        &alpha,
        static_cast<double*>(input.data()), batch_size,
        static_cast<double*>(weight.data()), dims_out,
        &beta,
        static_cast<double*>(output.data()), batch_size
    );
}

void backward_op_matmul(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncCudaDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous(device);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, input.shape(), device);
        input_grad.zero();
    }
    if (!weight_grad) {
        weight_grad = Tensor(DataType::dt_float, weight.shape(), device);
        weight_grad.zero();
    }
    if (!bias_grad) {
        bias_grad = Tensor(DataType::dt_float, {1, dims_out}, device);
        bias_grad.zero();
    }
    if (batch_size == 0) return;

    double alpha = 1., beta = 1.;
    cublasHandle_t handle = instruction.runtime.cublas_handle();
    cudaStream_t stream = device.stream();
    check_error(cublasSetStream(handle, stream));

    // compute input_grad += output_grad * weight
    check_error(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        batch_size, dims_in, dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()), batch_size,
        static_cast<double*>(weight.data()), dims_out,
        &beta,
        static_cast<double*>(input_grad.data()), batch_size
    ));

    // compute weight_grad += output_grad.T * input
    check_error(cublasDgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dims_out, dims_in, batch_size,
        &alpha,
        static_cast<double*>(output_grad.data()), batch_size,
        static_cast<double*>(input.data()), batch_size,
        &beta,
        static_cast<double*>(weight_grad.data()), dims_out
    ));

    // compute bias_grad += sum_i output_grad_ij
    double* ones;
    check_error(cudaMallocAsync(&ones, batch_size * sizeof(double), stream));
    thrust::fill_n(thrust::cuda::par.on(stream), thrust::device_pointer_cast(ones), batch_size, 1.0);
    check_error(cublasDgemv(
        handle,
        CUBLAS_OP_T,
        batch_size, dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()), batch_size,
        static_cast<double*>(ones), 1,
        &beta,
        static_cast<double*>(bias_grad.data()), 1
    ));
}

struct NotMinusOne {
    __device__ bool operator()(int64_t val) {
        return val != -1;
    }
};

__global__ void kernel_nonzero(
    std::size_t batch_size,
    CudaTensorView<double, 1, true> input,
    CudaTensorView<int64_t, 1, true> output
) {
    int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        output[i] = input[i] == 0. ? -1 : i;
    }
}

void op_nonzero(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor output_tmp(DataType::dt_int, {batch_size}, device);
    launch_kernel(
        kernel_nonzero, batch_size, device.stream(),
        batch_size, input.view<double, 1>(), output_tmp.view<int64_t, 1>()
    );

    auto input_ptr = thrust::device_pointer_cast(
        static_cast<int64_t*>(input.data())
    );
    auto output_ptr = thrust::device_pointer_cast(
        static_cast<int64_t*>(output_tmp.data())
    );
    cudaStreamSynchronize(device.stream());
    auto count = thrust::copy_if(
        input_ptr, input_ptr + batch_size, output_ptr, NotMinusOne()
    ) - output_ptr;
    output = output_tmp.slice(0, 0, count);
}

template<int dim>
__global__ void batch_gather_kernel(
    std::size_t batch_size,
    CudaTensorView<int64_t, 1, true> indices,
    CudaTensorView<double, dim, true> values,
    CudaTensorView<double, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<CudaTypes>, dim-1>(values[indices[i]], selection[i]);
    }
}

template<int dim>
void batch_gather_impl(
    Tensor& indices, Tensor& values, Tensor& selection, const AsyncCudaDevice& device
) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;
    selection = Tensor(DataType::dt_float, out_shape, device);

    launch_kernel(
        batch_gather_kernel<dim>,
        batch_size,
        device.stream(),
        batch_size,
        indices.view<int64_t, 1>(),
        values.view<double, dim>(),
        selection.view<double, dim>()
    );
}

void op_batch_gather(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    //TODO: this only accidentally works for types other than double
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    switch (values.shape().size()) {
        case 1: batch_gather_impl<1>(indices, values, selection, device); break;
        case 2: batch_gather_impl<2>(indices, values, selection, device); break;
        case 3: batch_gather_impl<3>(indices, values, selection, device); break;
        case 4: batch_gather_impl<4>(indices, values, selection, device); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template<int dim>
__global__ void scatter_kernel(
    std::size_t batch_size,
    CudaTensorView<int64_t, 1, true> indices,
    CudaTensorView<double, dim, true> source,
    CudaTensorView<double, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<CudaTypes>, dim-1>(source[i], output[indices[i]]);
    }
}

template<int dim>
void scatter_impl(
    Tensor& indices, Tensor& source, Tensor& output, const AsyncCudaDevice& device
) {
    auto batch_size = indices.size(0);
    launch_kernel(
        scatter_kernel<dim>,
        batch_size,
        device.stream(),
        batch_size,
        indices.view<int64_t, 1>(),
        source.view<double, dim>(),
        output.view<double, dim>()
    );
}

void op_scatter(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy(device);
    switch (target.shape().size()) {
        case 1: scatter_impl<1>(indices, source, output, device); break;
        case 2: scatter_impl<2>(indices, source, output, device); break;
        case 3: scatter_impl<3>(indices, source, output, device); break;
        case 4: scatter_impl<4>(indices, source, output, device); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

void op_random(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(DataType::dt_float, {batch_size, dim}, device);
    curandGenerator_t generator = instruction.runtime.curand_generator();
    check_error(curandSetStream(generator, device.stream()));
    check_error(curandGenerateUniformDouble(
        generator, static_cast<double*>(output.data()), batch_size * dim
    ));
}

__global__ void kernel_unweight(
    std::size_t batch_size,
    CudaTensorView<double, 1, true> rand_in,
    CudaTensorView<double, 1, true> weights_in,
    CudaTensorView<double, 1, true> max_weights_in,
    CudaTensorView<double, 1, true> weights_out,
    CudaTensorView<int64_t, 1, true> indices_out
) {
    int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) return;

    auto rand = rand_in[i], weight = weights_in[i], max_weight = max_weights_in[i];
    bool accepted = max_weight * rand < weight;
    auto weight_clipped = weight < max_weight ? max_weight : weight;
    weights_out[i] = accepted ? weight_clipped : 0.;
    indices_out[i] = accepted ? i : -1;
}

void op_unweight(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];
    auto batch_size = weights.size(0);
    cudaStream_t stream = device.stream();

    Tensor rand(DataType::dt_float, {batch_size}, device);
    curandGenerator_t generator = instruction.runtime.curand_generator();
    check_error(curandSetStream(generator, stream));
    check_error(curandGenerateUniformDouble(
        generator, static_cast<double*>(rand.data()), batch_size
    ));

    Tensor indices_tmp(DataType::dt_int, {batch_size}, device);
    Tensor uw_weights_tmp(DataType::dt_float, {batch_size}, device);
    launch_kernel(
        kernel_unweight,
        batch_size,
        stream,
        batch_size,
        rand.view<double, 1>(),
        weights.view<double, 1>(),
        max_weight.view<double, 1>(),
        uw_weights_tmp.view<double, 1>(),
        indices_tmp.view<int64_t, 1>()
    );

    Tensor indices_compacted(DataType::dt_int, {batch_size}, device);
    cudaStreamSynchronize(stream);
    auto ptr_all = thrust::device_pointer_cast(
        static_cast<int64_t*>(indices_tmp.data())
    );
    auto ptr_compacted = thrust::device_pointer_cast(
        static_cast<int64_t*>(indices_compacted.data())
    );
    auto ptr_compacted_end = thrust::copy_if(
        ptr_all, ptr_all + batch_size, ptr_compacted, NotMinusOne()
    );

    std::size_t count = ptr_compacted_end - ptr_compacted;
    indices = indices_compacted.slice(0, 0, count);
    uw_weights = Tensor(DataType::dt_float, {count}, device);
    auto ptr_all_weights = thrust::device_pointer_cast(
        static_cast<double*>(uw_weights_tmp.data())
    );
    auto ptr_uw_weights = thrust::device_pointer_cast(
        static_cast<double*>(uw_weights.data())
    );
    thrust::gather(
        thrust::cuda::par.on(stream),
        ptr_compacted, ptr_compacted_end,
        ptr_all_weights, ptr_uw_weights
    );
}

}

CudaRuntime::CudaRuntime(const Function& function, ContextPtr context) :
    _context(context), input_count(function.inputs().size())
{
    check_error(curandCreateGenerator(&_curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
    std::random_device rand_dev;
    check_error(curandSetPseudoRandomGeneratorSeed(_curand_generator, rand_dev()));
    check_error(cublasCreate(&_cublas_handle));

    locals_init.resize(function.locals().size());
    requires_grad_init.resize(function.locals().size());
    LastUseOfLocals last_use(function);
    InstructionDependencies dependencies(function);

    std::size_t instr_index = 0;
    std::vector<int> forward_streams;
    std::vector<int> backward_streams;
    std::vector<int> local_sources(function.locals().size(), -1);
    for (auto& instr : function.instructions()) {
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        int forward_stream_index = -1, backward_stream_index = -1;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
            int local_source = local_sources.at(in.local_index);
            if (local_source != -1) {
                if (forward_streams.at(local_source)) {

                }

            }
        }
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
            local_sources.at(out.local_index) = instr_index;
        }

        if (forward_stream_index >= streams.size() || backward_stream_index >= streams.size()) {
            cudaStream_t new_stream;
            check_error(cudaStreamCreate(&new_stream));
            streams.push_back(new_stream);
        }

        instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *this,
            instr.instruction->differentiable(),
            streams.at(forward_stream_index),
            streams.at(backward_stream_index),
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({
                -1, {local_index}, {}, {}, {}, 0, *this, false
            });
        }
        ++instr_index;
    }

    for (auto& [name, value] : function.globals()) {
        Tensor global = context->global(name);
        auto& global_shape = value.type.shape;
        Sizes full_shape(global_shape.size() + 1);
        full_shape[0] = 1;
        std::copy(global_shape.begin(), global_shape.end(), full_shape.begin() + 1);
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            throw std::invalid_argument(std::format(
                "Global {} has wrong dtype or shape", name
            ));
        }
        locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            requires_grad_init.at(value.local_index) = true;
            grad_global_indices.push_back({name, value.local_index});
        }
    }

    for (auto& local : function.locals()) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, &CudaDevice::instance());
                locals_init[local.local_index] = tensor;
            },
            [](std::monostate val){}
        }, local.literal_value);
    }

    for (auto& out : function.outputs()) {
        output_indices.push_back(out.local_index);
    }
}

CudaRuntime::~CudaRuntime() {
    check_error(curandDestroyGenerator(_curand_generator));
    check_error(cublasDestroy(_cublas_handle));
    for (auto event : events) {
        cudaEventDestroy(event);
    }
    for (auto stream : streams) {
        cudaStreamDestroy(stream);
    }
}

TensorVec CudaRuntime::run(const TensorVec& inputs) const {
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        AsyncCudaDevice device(instr.stream);
        for (auto event : instr.wait_events) {
            check_error(cudaStreamWaitEvent(instr.stream, event));
        }
        switch (instr.opcode) {
            case -1: // free memory
                locals[instr.input_indices[0]].reset(device);
                break;
#include "runtime_mixin.h"
        }
        if (instr.record_event) {
            check_error(cudaEventRecord(instr.record_event, instr.stream));
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(cudaDeviceSynchronize());
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CudaRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto locals = locals_init;
    auto requires_grad = requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto [instr, instr_eval_grad] : zip(instructions, eval_grad)) {
        AsyncCudaDevice device(instr.stream);
        if (instr.differentiable) {
            for (auto input_index : instr.input_indices) {
                if (requires_grad[input_index]) {
                    instr_eval_grad = true;
                    break;
                }
            }
            if (instr_eval_grad) {
                //TODO: only store necessary
                for (auto input_index : instr.input_indices) {
                    store_local[input_index] = true;
                }
                for (auto output_index : instr.output_indices) {
                    store_local[output_index] = true;
                    requires_grad[output_index] = true;
                }
            }
        }
        for (auto event : instr.wait_events) {
            check_error(cudaStreamWaitEvent(instr.stream, event));
        }
        switch (instr.opcode) {
            case -1: { // free memory
                auto input_index = instr.input_indices[0];
                if (!store_local[input_index]) {
                    locals[input_index].reset(device);
                }
                break;
            }
#include "runtime_mixin.h"
        }
        if (instr.record_event) {
            check_error(cudaEventRecord(instr.record_event, instr.stream));
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(cudaDeviceSynchronize());
    return {outputs, locals, eval_grad};
}

std::tuple<
    TensorVec, std::vector<std::tuple<std::string, Tensor>>
> CudaRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) const {
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (
        auto [instr, instr_eval_grad] :
        zip(std::views::reverse(instructions), std::views::reverse(eval_grad))
    ) {
        if (!instr_eval_grad) continue;
        bool needs_grad = true;
        for (auto [output_index, output_dtype] : zip(
            instr.output_indices, instr.output_dtypes
        )) {
            if (!local_grads[output_index] && output_dtype == DataType::dt_float) {
                needs_grad = false;
                break;
            }
        }
        for (auto event : instr.backward_wait_events) {
            check_error(cudaStreamWaitEvent(instr.backward_stream, event));
        }
        if (needs_grad) {
            AsyncCudaDevice device(instr.backward_stream);
            switch (instr.opcode) {
#include "runtime_backward_mixin.h"
            }
        }
        if (instr.backward_record_event) {
            check_error(cudaEventRecord(instr.backward_record_event, instr.backward_stream));
        }
    }
    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    check_error(cudaDeviceSynchronize());
    return {{local_grads.begin(), local_grads.begin() + input_count}, global_grads};
}

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context) {
    return new CudaRuntime(function, context);
}
