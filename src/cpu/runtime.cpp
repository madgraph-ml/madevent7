#include "madevent/cpu/runtime.h"
#include "madevent/madcode/optimizer.h"
#include "madevent/util.h"

#include "../kernels/kernels.h"

#include <tuple>
#include <array>
#include <algorithm>
#include <ranges>

extern "C" void dgemm_(
    char* transa, char* transb,
    int* m, int* n, int* k,
    double* alpha,
    double* a, int* lda,
    double* b, int* ldb,
    double* beta,
    double* c, int* ldc
);

extern "C" void dgemv_(
    char* trans,
    int* m, int* n,
    double* alpha,
    double* a, int* lda,
    double* x, int* incx,
    double* beta,
    double* y, int* incy
);

using namespace madevent;
using namespace madevent_cpu;
using namespace madevent_kernels;

namespace {

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void batch_foreach(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::array<const Tensor*, n_in> inputs;
    for (int i = 0; i < n_in; ++i) {
        inputs[i] = &locals[instruction.input_indices[i]];
    }

    std::array<Tensor*, n_out> outputs;
    for (int i = 0; i < n_out; ++i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        Sizes shape(output_shape.size() + 1);
        shape[0] = batch_size;
        std::copy(output_shape.begin(), output_shape.end(), shape.begin() + 1);
        output = Tensor(instruction.output_dtypes[i], shape);
        outputs[i] = &output;
    }

    if constexpr (dims == 0) {
        tensor_foreach_dynamic<scalar_func, vector_func, n_in, n_out>(
            inputs, outputs, batch_size
        );
    } else {
        tensor_foreach<scalar_func, vector_func, n_in, n_out, dims>(
            inputs, outputs, batch_size
        );
    }
}

template<
    auto scalar_func, auto vector_func,
    int n_in, int n_out,
    int n_in_stored, int n_out_stored,
    int dims
>
void backward_batch_foreach(
    const Runtime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    std::array<std::size_t, n_in_stored> in_stored_indices,
    std::array<std::size_t, n_out_stored> out_stored_indices
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    constexpr int n_args = n_in_stored + n_out_stored + n_out;
    std::array<const Tensor*, n_args> args;
    std::array<Tensor*, n_in> input_grads;
    for (int i = 0; i < n_in_stored; ++i) {
        args[i] = &locals[instruction.input_indices[in_stored_indices[i]]];
    }
    for (int i = 0; i < n_out_stored; ++i) {
        args[n_in_stored + i] = &locals[instruction.output_indices[out_stored_indices[i]]];
    }
    for (int i = 0; i < n_out; ++i) {
        args[n_in_stored + n_out_stored + i] = &local_grads[instruction.output_indices[i]];
    }
    for (int i = 0; i < n_in; ++i) {
        auto input_index = instruction.input_indices[i];
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            auto& input = locals[input_index];
            //auto& grad_shape = input.shape();
            //Sizes shape(grad_shape.size() + 1);
            //shape[0] = batch_size;
            //std::copy(grad_shape.begin(), grad_shape.end(), shape.begin() + 1);
            input_grad = Tensor(input.dtype(), input.shape());
            input_grad.zero();
        }
        input_grads[i] = &input_grad;
    }

    if constexpr (dims == 0) {
        tensor_foreach_dynamic<scalar_func, vector_func, n_args, n_in>(
            args, input_grads, batch_size
        );
    } else {
        tensor_foreach<scalar_func, vector_func, n_args, n_in, dims>(
            args, input_grads, batch_size
        );
    }
}

void op_stack(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto& first_shape = locals[instruction.input_indices[0]].shape();
    Sizes shape(first_shape.size() + 1);
    shape[0] = locals[instruction.batch_size_index].size(0);
    shape[1] = instruction.input_indices.size();
    std::copy(first_shape.begin() + 1, first_shape.end(), shape.begin() + 2);
    Tensor output(instruction.output_dtypes.front(), shape);
    std::size_t index = 0;
    for (auto input_index : instruction.input_indices) {
        output.select(1, index).copy_from(locals[input_index]);
        ++index;
    }
    locals[instruction.output_indices[0]] = output;
}

void backward_op_stack(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    // TODO: differentiate integer and float here (also other backwards)
    auto& output_grad = local_grads[instruction.output_indices[0]];
    std::size_t index = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
            input_grad.zero();
        }
        input_grad.add(output_grad.select(1, index));
        ++index;
    }
}

void op_unstack(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto tensors = locals[instruction.input_indices[0]].unstack(1);
    for (
        auto [tensor, output_index] :
        std::views::zip(tensors, instruction.output_indices)
    ) {
        locals[output_index] = tensor;
    }
}

void backward_op_unstack(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto input_index = instruction.input_indices[0];
    auto& input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
        input_grad.zero();
    }
    auto unstacked_grads = input_grad.unstack(1);
    for (
        auto [grad, output_index] :
        std::views::zip(unstacked_grads, instruction.output_indices)
    ) {
        grad.add(local_grads[output_index]);
    }
}

void op_pop(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto input = locals[instruction.input_indices[0]];
    std::size_t last_index = input.size(1) - 1;
    locals[instruction.output_indices[0]] = input.slice(1, 0, last_index);
    locals[instruction.output_indices[1]] = input.select(1, last_index);
}

void backward_op_pop(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto input_index = instruction.input_indices[0];
    auto input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
        input_grad.zero();
    }
    std::size_t last_index = input_grad.size(1) - 1;
    input_grad.slice(1, 0, last_index).add(local_grads[instruction.output_indices[0]]);
    input_grad.select(1, last_index).add(local_grads[instruction.output_indices[1]]);
}

void op_batch_cat(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = 0;
    SizeVec sizes;
    for (auto input_index : instruction.input_indices) {
        auto size = locals[input_index].size(0);
        sizes.push_back(size);
        batch_size += size;
    }
    auto shape = locals[instruction.input_indices.front()].shape();
    shape[0] = batch_size;
    Tensor output(instruction.output_dtypes.front(), shape);
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input = locals[input_index];
        auto next_offset = offset + input.size(0);
        output.slice(0, offset, next_offset).copy_from(input);
        offset = next_offset;
    }

    locals[instruction.output_indices[0]] = output;
    locals[instruction.output_indices[1]] = Tensor(sizes);
}

void backward_op_batch_cat(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto output_grad = local_grads[instruction.output_indices[0]];
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        auto next_offset = offset + input_grad.size(0);
        if (!input_grad) {
            input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
            input_grad.zero();
        }
        input_grad.add(output_grad.slice(0, offset, next_offset));
        offset = next_offset;
    }
}

void op_batch_split(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto tensors = locals[instruction.input_indices[0]].split(0, sizes);
    for (auto [tensor, output_index] : std::views::zip(tensors, instruction.output_indices)) {
        locals[output_index] = tensor;
    }
}

void backward_op_batch_split(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto input_index = instruction.input_indices[0];
    auto& input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
        input_grad.zero();
    }
    auto split_grads = input_grad.split(0, sizes);
    for (
        auto [tensor, output_index] :
        std::views::zip(split_grads, instruction.output_indices)
    ) {
        tensor.add(locals[output_index]);
    }

}

void op_cat(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t cat_size = 0;
    for (auto input_index : instruction.input_indices) {
        cat_size += locals[input_index].size(1);
    }

    auto& first_shape = locals[instruction.input_indices[0]].shape();
    Sizes shape(first_shape.size());
    shape[0] = locals[instruction.batch_size_index].size(0);
    shape[1] = cat_size;
    std::copy(first_shape.begin() + 2, first_shape.end(), shape.begin() + 2);

    Tensor output(instruction.output_dtypes.front(), shape);
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input = locals[input_index];
        auto next_offset = offset + input.size(1);
        output.slice(1, offset, next_offset).copy_from(input);
        offset = next_offset;
    }
    locals[instruction.output_indices[0]] = output;
}

void backward_op_cat(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto& output_grad = local_grads[instruction.output_indices[0]];
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            input_grad = Tensor(DataType::dt_float, locals[input_index].shape());
            input_grad.zero();
        }
        auto next_offset = offset + input_grad.size(1);
        input_grad.add(output_grad.slice(1, offset, next_offset));
        offset = next_offset;
    }
}

void op_matrix_element(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size});
    std::size_t me_index = locals[instruction.input_indices[1]].view<int64_t, 0>();
    instruction.context.matrix_element(me_index).call(
        locals[instruction.input_indices[0]], me_out
    );
}

void op_matrix_element_multichannel(
    const Runtime::Instruction& instruction, TensorVec& locals
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size});
    auto& chan_weights_out = locals[instruction.output_indices[1]];
    std::size_t me_index = locals[instruction.input_indices[2]].view<int64_t, 0>();
    std::size_t channel_count = locals[instruction.input_indices[3]].view<int64_t, 0>();
    chan_weights_out = Tensor(DataType::dt_float, {batch_size, channel_count});
    instruction.context.matrix_element(me_index).call_multichannel(
        locals[instruction.input_indices[0]],
        locals[instruction.input_indices[1]],
        me_out,
        chan_weights_out
    );
}

void op_pdf(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(DataType::dt_float, {batch_size});
    instruction.context.pdf_set().call(
        locals[instruction.input_indices[0]],
        locals[instruction.input_indices[1]],
        locals[instruction.input_indices[2]],
        output
    );
}

void op_matmul(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto input = locals[instruction.input_indices[0]].contiguous();
    auto weight = locals[instruction.input_indices[1]].contiguous();
    auto bias = locals[instruction.input_indices[2]].contiguous();
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(DataType::dt_float, {batch_size, dims_out});
    output.copy_from(bias);

    char transa = 'N', transb = 'T';
    int m = batch_size, n = dims_out, k = dims_in;
    double alpha = 1., beta = 1.;
    int lda = batch_size, ldb = dims_out, ldc = batch_size;
    double* a = static_cast<double*>(input.data());
    double* b = static_cast<double*>(weight.data());
    double* c = static_cast<double*>(output.data());
    dgemm_(
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
    );
}

void backward_op_matmul(
    const Runtime::Instruction& instruction, TensorVec& locals, TensorVec& local_grads
) {
    auto input = locals[instruction.input_indices[0]].contiguous();
    auto weight = locals[instruction.input_indices[1]].contiguous();
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous();
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, input.shape());
        input_grad.zero();
    }
    if (!weight_grad) {
        weight_grad = Tensor(DataType::dt_float, weight.shape());
        weight_grad.zero();
    }
    if (!bias_grad) {
        bias_grad = Tensor(DataType::dt_float, {1, dims_out});
        bias_grad.zero();
    }

    // compute input_grad += output_grad * weight
    {
        char transa = 'N', transb = 'N';
        int m = batch_size, n = dims_in, k = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = dims_out, ldc = batch_size;
        double* a = static_cast<double*>(output_grad.data());
        double* b = static_cast<double*>(weight.data());
        double* c = static_cast<double*>(input_grad.data());
        dgemm_(
            &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
        );
    }

    // compute weight_grad += output_grad.T * input
    {
        char transa = 'T', transb = 'N';
        int m = dims_out, n = dims_in, k = batch_size;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = batch_size, ldc = dims_out;
        double* a = static_cast<double*>(output_grad.data());
        double* b = static_cast<double*>(input.data());
        double* c = static_cast<double*>(weight_grad.data());
        dgemm_(
            &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
        );
    }

    // compute bias_grad += sum_i output_grad_ij
    {
        // TODO: we should probably do this differently...
        std::vector<double> ones(batch_size, 1.);
        char trans = 'T';
        int m = batch_size, n = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, incx = 1, incy = 1;
        double* a = static_cast<double*>(output_grad.data());
        double* x = ones.data();
        double* y = static_cast<double*>(bias_grad.data());
        dgemv_(
            &trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy
        );
    }
}

void op_nonzero(const Runtime::Instruction& instruction, TensorVec& locals) {
    // TODO: not parallelized for now
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor output_tmp(DataType::dt_int, {batch_size});
    auto input_view = input.view<double, 1>();
    auto output_view = output_tmp.view<int64_t, 1>();
    std::size_t count = 0;
    for (std::size_t i = 0; i < batch_size; ++i) {
        if (input_view[i] != 0.) {
            output_view[count] = i;
            ++count;
        }
    }
    output = output_tmp.slice(0, 0, count);
}

template<int dim>
void batch_gather_impl(Tensor& indices, Tensor& values, Tensor& selection) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;
    selection = Tensor(DataType::dt_float, out_shape);
    auto indices_view = indices.view<int64_t, 1>();
    auto values_view = values.view<double, dim>();
    auto selection_view = selection.view<double, dim>();
    ThreadPool::instance().parallel_for([&](std::size_t i) {
        recursive_for<kernel_copy<CpuTypes>, dim-1>(
            values_view[indices_view[i]], selection_view[i]
        );
    }, batch_size);
}

void op_batch_gather(const Runtime::Instruction& instruction, TensorVec& locals) {
    //TODO: this only accidentally works for types other than double
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    switch (values.shape().size()) {
        case 1: batch_gather_impl<1>(indices, values, selection); break;
        case 2: batch_gather_impl<2>(indices, values, selection); break;
        case 3: batch_gather_impl<3>(indices, values, selection); break;
        case 4: batch_gather_impl<4>(indices, values, selection); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template<int dim>
void scatter_impl(Tensor& indices, Tensor& source, Tensor& output) {
    auto batch_size = indices.size(0);
    auto indices_view = indices.view<int64_t, 1>();
    auto source_view = source.view<double, dim>();
    auto output_view = output.view<double, dim>();
    ThreadPool::instance().parallel_for([&](std::size_t i) {
        recursive_for<kernel_copy<CpuTypes>, dim-1>(
            source_view[i], output_view[indices_view[i]]
        );
    }, batch_size);
}

void op_scatter(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy();
    switch (target.shape().size()) {
        case 1: scatter_impl<1>(indices, source, output); break;
        case 2: scatter_impl<2>(indices, source, output); break;
        case 3: scatter_impl<3>(indices, source, output); break;
        case 4: scatter_impl<4>(indices, source, output); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

void op_random(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(DataType::dt_float, {batch_size, dim});
    auto output_view = output.view<double, 1>();
    auto& pool = ThreadPool::instance();
    pool.parallel_for<ThreadPool::pass_thread_id>(
        [&](std::size_t i, std::size_t thread_id
    ) {
        output_view[i] = pool.random(thread_id);
    }, batch_size * dim);
}

void op_unweight(const Runtime::Instruction& instruction, TensorVec& locals) {
    // TODO: not parallelized for now
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];

    auto batch_size = weights.size(0);
    Tensor indices_tmp(DataType::dt_int, {batch_size});
    Tensor uw_weights_tmp(DataType::dt_float, {batch_size});

    auto weights_view = weights.view<double, 1>();
    double max_weight_val = max_weight.view<double, 0>();
    auto indices_view = indices_tmp.view<int64_t, 1>();
    auto uw_weights_view = uw_weights_tmp.view<double, 1>();
    auto& pool = ThreadPool::instance();

    std::size_t count = 0;
    for (std::size_t i = 0; i < batch_size; ++i) {
        double w = weights_view[i];
        if (w != 0. && w / max_weight_val > pool.random(0)) {
            indices_view[count] = i;
            uw_weights_view[count] = w > max_weight_val ? w : max_weight_val;
            ++count;
        }
    }

    indices = indices_tmp.slice(0, 0, count);
    uw_weights = uw_weights_tmp.slice(0, 0, count);
}

}

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    //TODO: this function is in the wrong place. need some restructuring
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
        {&source}, {&target}, target.size(0)
    );
}

void CpuDevice::tensor_zero(Tensor& tensor) const {
    //TODO: this function is in the wrong place. need some restructuring
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_zero<CpuTypes>, kernel_zero<SimdTypes>, 1, 1>(
        {&tensor}, {&tensor}, tensor.size(0)
    );
}

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    //TODO: this function is in the wrong place. need some restructuring
    tensor_foreach_dynamic<kernel_add_inplace<CpuTypes>, kernel_add_inplace<SimdTypes>, 1, 1>(
        {&source}, {&target}, target.size(0)
    );
}

Runtime::Runtime(const Function& function, ContextPtr context) :
    context(context), input_count(function.inputs().size())
{
    locals_init.resize(function.locals().size());
    requires_grad_init.resize(function.locals().size());
    auto opt_function = optimize_constants(function);
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(opt_function);

    for (auto& instr : opt_function.instructions()) {
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
        }
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
        }
        instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *context,
            instr.instruction->differentiable()
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({
                -1, {local_index}, {}, {}, {}, 0, *context, false
            });
        }
        ++instr_index;
    }

    for (auto& [name, value] : opt_function.globals()) {
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

    for (auto& local : opt_function.locals()) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, cpu_device());
                locals_init[local.local_index] = tensor;
            },
            [&](TensorValue val) {
                auto& [shape, items] = val;
                Sizes full_shape(shape.size() + 1);
                full_shape[0] = 1;
                std::copy(shape.begin(), shape.end(), full_shape.begin() + 1);
                Tensor tensor(local.type.dtype, full_shape);
                std::visit([&](auto items) {
                    auto tview = tensor.template view<
                        typename decltype(items)::value_type, 2>()[0];
                    std::size_t i = 0;
                    for (auto item : items) {
                        tview[i] = item;
                        ++i;
                    }
                }, items);
                locals_init[local.local_index] = tensor;
            },
            [](std::string val){},
            [](std::monostate val){}
        }, local.literal_value);
    }

    for (auto& out : opt_function.outputs()) {
        output_indices.push_back(out.local_index);
    }
}

TensorVec Runtime::run(const TensorVec& inputs) const {
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        switch (instr.opcode) {
            case -1: // free memory
                locals[instr.input_indices[0]].reset();
                break;
#include "runtime_mixin.h"
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> Runtime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto locals = locals_init;
    auto requires_grad = requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto [instr, instr_eval_grad] : std::views::zip(instructions, eval_grad)) {
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
        switch (instr.opcode) {
            case -1: { // free memory
                auto input_index = instr.input_indices[0];
                if (!store_local[input_index]) {
                    locals[input_index].reset();
                }
                break;
            }
#include "runtime_mixin.h"
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    return {outputs, locals, eval_grad};
}

std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>> Runtime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) {
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : std::views::zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (
        auto [instr, instr_eval_grad] :
        std::views::reverse(std::views::zip(instructions, eval_grad))
    ) {
        if (!instr_eval_grad) continue;
        bool needs_grad = true;
        for (auto [output_index, output_dtype] : std::views::zip(
            instr.output_indices, instr.output_dtypes
        )) {
            if (!local_grads[output_index] && output_dtype == DataType::dt_float) {
                needs_grad = false;
                break;
            }
        }
        if (needs_grad) {
            switch (instr.opcode) {
#include "runtime_backward_mixin.h"
            }
        }
    }
    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    return {{local_grads.begin(), local_grads.begin() + input_count}, global_grads};
}

