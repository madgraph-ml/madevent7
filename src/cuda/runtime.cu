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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

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
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size}, device);
    std::size_t me_index = locals[instruction.input_indices[2]].index_value();
    auto& matrix_element = instruction.runtime.context().matrix_element(me_index);
    // TODO: maybe copy can be avoided sometimes
    auto momenta_in = locals[instruction.input_indices[0]].contiguous(batch_size, device);
    auto flavor_in = locals[instruction.input_indices[1]].contiguous(batch_size, device);

    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != matrix_element.particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    if (!matrix_element.on_gpu()) {
        throw std::runtime_error("Incompatible device");
    }
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto flavor_ptr = static_cast<me_int_t*>(flavor_in.data());
    auto me_ptr = static_cast<double*>(me_out.data());
    matrix_element.call(
        matrix_element.process_instance(ThreadPool::thread_index()),
        batch_size, batch_size, mom_ptr, flavor_ptr, me_ptr,
        device.stream()
    );
}

void op_matrix_element_multichannel(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t me_index = locals[instruction.input_indices[4]].index_value();
    std::size_t diagram_count = locals[instruction.input_indices[5]].index_value();

    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size}, device);
    auto& amp2_out = locals[instruction.output_indices[1]];
    amp2_out = Tensor(DataType::dt_float, {batch_size, diagram_count}, device);
    auto& diagram_out = locals[instruction.output_indices[2]];
    diagram_out = Tensor(DataType::dt_int, {batch_size}, device);
    auto& color_out = locals[instruction.output_indices[3]];
    color_out = Tensor(DataType::dt_int, {batch_size}, device);
    auto& helicity_out = locals[instruction.output_indices[4]];
    helicity_out = Tensor(DataType::dt_int, {batch_size}, device);

    auto& matrix_element = instruction.runtime.context().matrix_element(me_index);

    // TODO: maybe copy can be avoided sometimes
    auto momenta_in = locals[instruction.input_indices[0]].contiguous(batch_size, device);
    auto alpha_s_in = locals[instruction.input_indices[1]].contiguous(batch_size, device);
    auto random_in = locals[instruction.input_indices[2]].contiguous(batch_size, device);
    auto flavor_in = locals[instruction.input_indices[3]].contiguous(batch_size, device);
    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != matrix_element.particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    if (diagram_count != matrix_element.diagram_count()) {
        throw std::runtime_error("Incompatible diagram count");
    }
    if (!matrix_element.on_gpu()) {
        throw std::runtime_error("Incompatible device");
    }

    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto alpha_ptr = static_cast<double*>(alpha_s_in.data());
    auto random_ptr = static_cast<double*>(random_in.data());
    auto flavor_ptr = static_cast<me_int_t*>(flavor_in.data());
    auto me_ptr = static_cast<double*>(me_out.data());
    auto amp2_ptr = static_cast<double*>(amp2_out.data());
    auto diag_ptr = static_cast<me_int_t*>(diagram_out.data());
    auto color_ptr = static_cast<me_int_t*>(color_out.data());
    auto helicity_ptr = static_cast<me_int_t*>(helicity_out.data());

    matrix_element.call_multichannel(
        matrix_element.process_instance(ThreadPool::thread_index()),
        batch_size, batch_size,
        mom_ptr, alpha_ptr, random_ptr, flavor_ptr, me_ptr,
        amp2_ptr, color_ptr, diag_ptr, helicity_ptr,
        device.stream()
    );
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
    check_error(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch_size, dims_out, dims_in,
        &alpha,
        static_cast<double*>(input.data()), batch_size,
        static_cast<double*>(weight.data()), dims_out,
        &beta,
        static_cast<double*>(output.data()), batch_size
    ));
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
    cudaFreeAsync(ones, stream);
}

struct NotMinusOne {
    __device__ bool operator()(me_int_t val) {
        return val != -1;
    }
};

__global__ void kernel_nonzero(
    std::size_t batch_size,
    CudaTensorView<double, 1, true> input,
    CudaTensorView<me_int_t, 1, true> output
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
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
    Tensor indices_tmp(DataType::dt_int, {batch_size}, device);
    Tensor output_tmp(DataType::dt_int, {batch_size}, device);
    launch_kernel(
        kernel_nonzero, batch_size, device.stream(),
        batch_size, input.view<double, 1>(), indices_tmp.view<me_int_t, 1>()
    );

    auto indices_ptr = thrust::device_pointer_cast(
        static_cast<me_int_t*>(indices_tmp.data())
    );
    auto output_ptr = thrust::device_pointer_cast(
        static_cast<me_int_t*>(output_tmp.data())
    );
    check_error(cudaStreamSynchronize(device.stream()));
    auto count = thrust::copy_if(
        indices_ptr, indices_ptr + batch_size, output_ptr, NotMinusOne()
    ) - output_ptr;
    output = output_tmp.slice(0, 0, count);
}

template<int dim>
__global__ void batch_gather_kernel(
    std::size_t batch_size,
    CudaTensorView<me_int_t, 1, true> indices,
    CudaTensorView<double, dim, true> values,
    CudaTensorView<double, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<CudaTypes>, dim-1>(values[indices[i]], selection[i]);
    }
}

template<int dim>
__global__ void batch_gather_kernel_int(
    std::size_t batch_size,
    CudaTensorView<me_int_t, 1, true> indices,
    CudaTensorView<me_int_t, dim, true> values,
    CudaTensorView<me_int_t, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<CudaTypes>, dim-1>(values[indices[i]], selection[i]);
    }
}

template<int dim>
void batch_gather_impl(
    Tensor& indices, Tensor& values, Tensor& selection, const AsyncCudaDevice& device
) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;

    if (values.dtype() == DataType::dt_float) {
        selection = Tensor(DataType::dt_float, out_shape, device);
        launch_kernel(
            batch_gather_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<double, dim>(),
            selection.view<double, dim>()
        );
    } else if (values.dtype() == DataType::dt_int) {
        selection = Tensor(DataType::dt_int, out_shape, device);
        launch_kernel(
            batch_gather_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<me_int_t, dim>(),
            selection.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_gather");
    }
}

void op_batch_gather(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
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
__global__ void batch_scatter_kernel(
    std::size_t batch_size,
    CudaTensorView<me_int_t, 1, true> indices,
    CudaTensorView<double, dim, true> source,
    CudaTensorView<double, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<CudaTypes>, dim-1>(source[i], output[indices[i]]);
    }
}

template<int dim>
__global__ void batch_scatter_kernel_int(
    std::size_t batch_size,
    CudaTensorView<me_int_t, 1, true> indices,
    CudaTensorView<me_int_t, dim, true> source,
    CudaTensorView<me_int_t, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<CudaTypes>, dim-1>(source[i], output[indices[i]]);
    }
}

template<int dim>
void batch_scatter_impl(
    Tensor& indices, Tensor& source, Tensor& output, const AsyncCudaDevice& device
) {
    if (source.dtype() == DataType::dt_float) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<double, dim>(),
            output.view<double, dim>()
        );
    } else if (source.dtype() == DataType::dt_int) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<me_int_t, dim>(),
            output.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_scatter");
    }
}

void op_batch_scatter(
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
        case 1: batch_scatter_impl<1>(indices, source, output, device); break;
        case 2: batch_scatter_impl<2>(indices, source, output, device); break;
        case 3: batch_scatter_impl<3>(indices, source, output, device); break;
        case 4: batch_scatter_impl<4>(indices, source, output, device); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

void op_offset_indices(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& sizes_offset = locals[instruction.input_indices[0]].batch_sizes();
    auto& sizes_out = locals[instruction.input_indices[1]].batch_sizes();
    std::size_t total_size = std::accumulate(sizes_out.begin(), sizes_out.end(), 0);
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(DataType::dt_int, {total_size}, device);
    std::size_t sum_offset = 0, sum_out = 0;
    for (auto [size_offset, size_out] : zip(sizes_offset, sizes_out)) {
        thrust::fill_n(
            thrust::cuda::par.on(device.stream()),
            thrust::device_pointer_cast(
                static_cast<me_int_t*>(output.data()) + sum_out
            ),
            size_out,
            sum_offset
        );
        sum_offset += size_offset;
        sum_out += size_out;
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
    CudaTensorView<me_int_t, 1, true> indices_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
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
        indices_tmp.view<me_int_t, 1>()
    );

    Tensor indices_compacted(DataType::dt_int, {batch_size}, device);
    check_error(cudaStreamSynchronize(stream));
    auto ptr_all = thrust::device_pointer_cast(
        static_cast<me_int_t*>(indices_tmp.data())
    );
    auto ptr_compacted = thrust::device_pointer_cast(
        static_cast<me_int_t*>(indices_compacted.data())
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

struct Decrement {
    __device__ void operator()(me_int_t& val) {
        --val;
    }
};

void histogram_common(
    const AsyncCudaDevice& device,
    std::size_t padded_size,
    std::size_t n_dims,
    std::size_t n_bins,
    Tensor& indices_tmp,
    Tensor& weights_tmp,
    Tensor& counts,
    Tensor& values
) {
    auto policy = thrust::cuda::par.on(device.stream());
    Tensor reduce_tmp(DataType::dt_float, {n_dims * n_bins}, device);
    auto indices_ptr = thrust::device_pointer_cast(
        static_cast<me_int_t*>(indices_tmp.data())
    );
    auto weights_ptr = thrust::device_pointer_cast(
        static_cast<double*>(weights_tmp.data())
    );
    auto counts_ptr = thrust::device_pointer_cast(
        static_cast<me_int_t*>(counts.data())
    );
    auto values_ptr = thrust::device_pointer_cast(
        static_cast<double*>(values.data())
    );
    auto reduce_tmp_ptr = thrust::device_pointer_cast(
        static_cast<double*>(reduce_tmp.data())
    );

    std::size_t flat_size = padded_size * n_dims;
    thrust::sort_by_key(policy, indices_ptr, indices_ptr + flat_size, weights_ptr);
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        thrust::constant_iterator<me_int_t>(1),
        reduce_tmp_ptr,
        counts_ptr
    );
    thrust::for_each_n(policy, counts_ptr, n_dims * n_bins, Decrement{});
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        weights_ptr,
        reduce_tmp_ptr,
        values_ptr
    );
}

__global__ void kernel_prepare_vegas_hist(
    std::size_t batch_size,
    std::size_t n_bins,
    CudaTensorView<double, 2, true> input,
    CudaTensorView<double, 1, true> weights_in,
    CudaTensorView<me_int_t, 2, true> indices,
    CudaTensorView<double, 2, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_bins) return;
    std::size_t n_dims = input.size(1);
    double bin_count_f = n_bins;
    double w2 = i < batch_size ? weights_in[i] * weights_in[i] : 0.;
    for (std::size_t j = 0; j < n_dims; ++j) {
        indices[i][j] = j + n_dims * (i < batch_size ?
            static_cast<me_int_t>(input[i][j] * bin_count_f) : i - batch_size);
        weights_out[i][j] = w2;
    }
}

void op_vegas_histogram(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_dims = input.size(1);
    std::size_t n_bins = values.size(2);

    std::size_t padded_size = batch_size + n_bins;
    Tensor indices_tmp(DataType::dt_int, {padded_size, n_dims}, device);
    Tensor weights_tmp(DataType::dt_float, {padded_size, n_dims}, device);
    launch_kernel(
        kernel_prepare_vegas_hist, padded_size, device.stream(), batch_size, n_bins,
        input.view<double, 2>(), weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 2>(), weights_tmp.view<double, 2>()
    );
    histogram_common(
        device, padded_size, n_dims, n_bins, indices_tmp, weights_tmp, counts, values
    );
}

__global__ void kernel_prepare_discrete_hist(
    std::size_t batch_size,
    std::size_t n_opts,
    CudaTensorView<me_int_t, 1, true> input,
    CudaTensorView<double, 1, true> weights_in,
    CudaTensorView<me_int_t, 1, true> indices,
    CudaTensorView<double, 1, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_opts) return;
    indices[i] = i < batch_size ? input[i] : i - batch_size;
    weights_out[i] = i < batch_size ? weights_in[i] : 0.;
}

void op_discrete_histogram(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_opts = values.size(1);

    std::size_t padded_size = batch_size + n_opts;
    Tensor indices_tmp(DataType::dt_int, {padded_size}, device);
    Tensor weights_tmp(DataType::dt_float, {padded_size}, device);
    launch_kernel(
        kernel_prepare_discrete_hist, padded_size, device.stream(), batch_size, n_opts,
        input.view<me_int_t, 1>(), weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>(), weights_tmp.view<double, 1>()
    );
    histogram_common(
        device, padded_size, 1, n_opts, indices_tmp, weights_tmp, counts, values
    );
}

}

CudaRuntime::CudaRuntime(const Function& function, ContextPtr context) :
    _context(context), _input_count(function.inputs().size())
{
    check_error(curandCreateGenerator(&_curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
    std::random_device rand_dev;
    check_error(curandSetPseudoRandomGeneratorSeed(_curand_generator, rand_dev()));
    check_error(cublasCreate(&_cublas_handle));

    InstructionDependencies dependencies(function);
    auto order = dependencies.ranks();
    SizeVec instruction_perm(function.instructions().size());
    std::iota(instruction_perm.begin(), instruction_perm.end(), 0);
    std::sort(
        instruction_perm.begin(), instruction_perm.end(),
        [&](std::size_t i, std::size_t j) { return order.at(i) < order.at(j); }
    );

    /*std::vector<int> local_sources_backward(function.locals().size(), -1);
    SizeVec stream_last_instr, instr_stream;
    for (std::size_t instr_index : std::views::reverse(instruction_perm)) {
        auto& instr = function.instructions().at(instr_index);
    }*/

    _locals_init.resize(function.locals().size());
    _requires_grad_init.resize(function.locals().size());
    LastUseOfLocals last_use(function);

    std::vector<int> local_sources(function.locals().size(), -1);
    SizeVec real_indices;
    std::vector<int> stream_last_instr, instr_streams;
    std::size_t event_index = 0;
    for (std::size_t instr_index_new = 0; std::size_t instr_index : instruction_perm) {
        auto& instr = function.instructions().at(instr_index);
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;

        int instr_stream = -1;
        std::vector<int> instr_deps;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }

            int local_source = local_sources.at(in.local_index);
            if (local_source == -1) continue;

            if (
                instr_stream == -1 &&
                stream_last_instr.at(instr_streams.at(local_source)) == local_source
            ) {
                instr_stream = instr_streams.at(local_source);
            } else {
                int local_source_perm = instruction_perm.at(local_source);
                instr_deps.erase(
                    std::remove_if(instr_deps.begin(), instr_deps.end(), [&](int i) {
                        return i == local_source || dependencies.depends(
                            local_source_perm, instruction_perm.at(i)
                        );
                    }),
                    instr_deps.end()
                );
                instr_deps.push_back(local_source);
            }
        }
        if (instr_stream == -1) {
            for (int i : stream_last_instr) {
                int dep_index = instruction_perm.at(i);
                if (dependencies.depends(instr_index_new, dep_index)) {
                    instr_stream = instr_streams.at(dep_index);
                    break;
                }
            }

            if (instr_stream == -1) {
                instr_stream = stream_last_instr.size();
                stream_last_instr.push_back(instr_index_new);
            }
        }
        stream_last_instr.at(instr_stream) = instr_index_new;
        instr_streams.push_back(instr_stream);
        for (int& dep : instr_deps) {
            auto& dep_instr = _instructions.at(real_indices.at(dep));
            if (dep_instr.record_event == -1) {
                dep_instr.record_event = event_index;
                ++event_index;
            }
            dep = dep_instr.record_event;
        }

        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
            local_sources.at(out.local_index) = instr_index_new;
        }

        real_indices.push_back(_instructions.size());
        _instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *this,
            instr.instruction->differentiable(),
            instr_stream,
            0,
            instr_deps,
            -1,
            {},
            -1
        });
        //for (std::size_t local_index : last_use.local_indices(instr_index)) {
        //    _instructions.push_back({
        //        -1, {local_index}, {}, {}, {}, 0, *this, false, 0, 0
        //    });
        //}
        ++instr_index_new;
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
        _locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            _requires_grad_init.at(value.local_index) = true;
            _grad_global_indices.push_back({name, value.local_index});
        }
    }

    for (auto& local : function.locals()) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, &CudaDevice::instance());
                _locals_init[local.local_index] = tensor;
            },
            [](std::monostate val){}
        }, local.literal_value);
    }

    std::vector<int> out_deps;
    for (std::size_t out_index = 0; auto& out : function.outputs()) {
        _output_indices.push_back(out.local_index);
        int local_source = local_sources.at(out.local_index);
        if (out_index == 0) {
            _sync_stream = instr_streams.at(local_source);
        } else {
            int local_source_perm = instruction_perm.at(local_source);
            out_deps.erase(
                std::remove_if(out_deps.begin(), out_deps.end(), [&](int i) {
                    return i == local_source || dependencies.depends(
                        local_source_perm, instruction_perm.at(i)
                    );
                }),
                out_deps.end()
            );
            out_deps.push_back(local_source);
        }
        ++out_index;
    }
    for (int& dep : out_deps) {
        auto& dep_instr = _instructions.at(real_indices.at(dep));
        if (dep_instr.record_event == -1) {
            dep_instr.record_event = event_index;
            ++event_index;
        }
        _sync_events.push_back(dep_instr.record_event);
    }

    _sync_stream = 0;
    _backward_sync_stream = 0;
    std::size_t event_count = event_index;
    std::size_t stream_count = stream_last_instr.size();
    println("stream count: {}, event count: {}", stream_count, event_count);

    _streams = ThreadResource<std::vector<cudaStream_t>>(
        default_thread_pool(),
        [stream_count]() {
            std::vector<cudaStream_t> streams(stream_count);
            for (auto& stream : streams) cudaStreamCreate(&stream);
            return streams;
        },
        [](auto& streams) {
            for (auto& stream : streams) cudaStreamDestroy(stream);
        }
    );
    _events = ThreadResource<std::vector<cudaEvent_t>>(
        default_thread_pool(),
        [event_count]() {
            std::vector<cudaEvent_t> events(event_count);
            for (auto& event : events) cudaEventCreate(&event);
            return events;
        },
        [](auto& events) {
            for (auto& event : events) cudaEventDestroy(event);
        }
    );
}

CudaRuntime::~CudaRuntime() {
    check_error(curandDestroyGenerator(_curand_generator));
    check_error(cublasDestroy(_cublas_handle));
}

TensorVec CudaRuntime::run(const TensorVec& inputs) const {
    auto& streams = _streams.get(ThreadPool::thread_index()); 
    auto& events = _events.get(ThreadPool::thread_index()); 
    auto locals = _locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    //println("----");
    for (auto& instr : _instructions) {
        auto stream = streams[instr.stream];
        AsyncCudaDevice device(stream);
        //print("opcode={}, stream={}, wait=[", instr.opcode, instr.stream);
        for (auto event : instr.wait_events) {
            check_error(cudaStreamWaitEvent(stream, events[event]));
            //print("{}, ", event);
        }
        //println("], record={}", instr.record_event);
        switch (instr.opcode) {
            case -1: // free memory
                locals[instr.input_indices[0]].reset(device);
                break;
#include "runtime_mixin.h"
        }
        if (instr.record_event != -1) {
            check_error(cudaEventRecord(events[instr.record_event], stream));
        }
    }

    auto sync_stream = streams[_sync_stream];
    //print("SYNC, stream={}, wait=[", _sync_stream);
    for (auto event : _sync_events) {
        check_error(cudaStreamWaitEvent(sync_stream, events[event]));
        //print("{}, ", event);
    }
    //println("]");
    check_error(cudaStreamSynchronize(sync_stream));

    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CudaRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto& streams = _streams.get(ThreadPool::thread_index()); 
    auto& events = _events.get(ThreadPool::thread_index()); 
    auto locals = _locals_init;
    auto requires_grad = _requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(_instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    std::copy(input_requires_grad.begin(), input_requires_grad.end(), requires_grad.begin());

    for (auto [instr, instr_eval_grad] : zip(_instructions, eval_grad)) {
        auto stream = streams[instr.stream];
        AsyncCudaDevice device(stream);
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
            check_error(cudaStreamWaitEvent(stream, events[event]));
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
        if (instr.record_event != -1) {
            check_error(cudaEventRecord(events[instr.record_event], stream));
        }
    }

    auto sync_stream = streams[_sync_stream];
    for (auto event : _sync_events) {
        check_error(cudaStreamWaitEvent(sync_stream, events[event]));
    }
    check_error(cudaStreamSynchronize(sync_stream));

    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    return {outputs, locals, eval_grad};
}

std::tuple<
    TensorVec, std::vector<std::tuple<std::string, Tensor>>
> CudaRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) const {
    auto& streams = _streams.get(ThreadPool::thread_index()); 
    auto& events = _events.get(ThreadPool::thread_index()); 
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(_output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (
        auto [instr, instr_eval_grad] :
        zip(std::views::reverse(_instructions), std::views::reverse(eval_grad))
    ) {
        if (!instr_eval_grad) continue;
        auto stream = streams[instr.backward_stream];
        AsyncCudaDevice device(stream);
        for (auto [output_index, output_dtype] : zip(
            instr.output_indices, instr.output_dtypes
        )) {
            auto& grad = local_grads[output_index];
            if (!grad && output_dtype == DataType::dt_float) {
                grad = Tensor(DataType::dt_float, locals[output_index].shape(), device);
                grad.zero(device);
            }
        }
        for (auto event : instr.backward_wait_events) {
            check_error(cudaStreamWaitEvent(stream, events[event]));
        }
        switch (instr.opcode) {
#include "runtime_backward_mixin.h"
        }
        if (instr.backward_record_event != -1) {
            check_error(cudaEventRecord(events[instr.backward_record_event], stream));
        }
    }

    auto sync_stream = streams[_backward_sync_stream];
    for (auto event : _backward_sync_events) {
        check_error(cudaStreamWaitEvent(sync_stream, events[event]));
    }
    check_error(cudaStreamSynchronize(sync_stream));

    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : _grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    return {{local_grads.begin(), local_grads.begin() + _input_count}, global_grads};
}

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    return new CudaRuntime(function, context);
}
