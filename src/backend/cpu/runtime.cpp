#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode/optimizer.h"
#include "madevent/util.h"
#include "kernels.h"

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

using namespace madevent;
using namespace madevent::cpu;

namespace {

void tensor_copy(Tensor source, Tensor target) {
    tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
        {&source}, {&target}, target.size(0)
    );
}

void tensor_zero(Tensor tensor) {
    tensor_foreach_dynamic<kernel_zero<CpuTypes>, kernel_zero<SimdTypes>, 1, 1>(
        {&tensor}, {&tensor}, tensor.size(0)
    );
}

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void batch_foreach(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::array<Tensor*, n_in> inputs;
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
        //SizeVec shape {batch_size};
        //shape.insert(shape.end(), output_shape.begin(), output_shape.end());
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
    std::array<Tensor*, n_args> args;
    std::array<Tensor*, n_in> input_grads;
    int i = 0;
    for (; i < n_in_stored; ++i) {
        args[i] = &locals[instruction.input_indices[in_stored_indices[i]]];
    }
    for (; i < n_in_stored + n_out_stored; ++i) {
        args[i] = &locals[instruction.output_indices[out_stored_indices[i]]];
    }
    for (; i < n_args; ++i) {
        args[i] = &local_grads[instruction.output_indices[i]];
    }
    for (i = 0; i < n_in; ++i) {
        auto& input_grad = locals[instruction.input_indices[i]];
        if (!input_grad) {
            auto& input = locals[instruction.input_indices[i]];
            auto& grad_shape = input.shape();
            Sizes shape(grad_shape.size() + 1);
            shape[0] = batch_size;
            std::copy(grad_shape.begin(), grad_shape.end(), shape.begin() + 1);
            input_grad = Tensor(input.dtype(), shape);
            tensor_zero(input_grad);
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
        tensor_copy(locals[input_index], output.select(1, index));
        ++index;
    }
    locals[instruction.output_indices[0]] = output;
}

void op_unstack(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto tensors = locals[instruction.input_indices[0]].unstack(1);
    auto output_index = instruction.output_indices.begin();
    for (auto& tensor : tensors) {
        locals[*output_index] = tensor;
        ++output_index;
    }
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
        tensor_copy(input, output.slice(0, offset, next_offset));
        offset = next_offset;
    }

    locals[instruction.output_indices[0]] = output;
    locals[instruction.output_indices[1]] = Tensor(sizes);
}

void op_batch_split(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto tensors = locals[instruction.input_indices[0]].split(0, sizes);
    auto output_index = instruction.output_indices.begin();
    for (auto [tensor, output_index] : std::views::zip(tensors, instruction.output_indices)) {
        locals[output_index] = tensor;
    }
}

void op_matrix_element(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size});
    auto& chan_weights_out = locals[instruction.output_indices[1]];
    chan_weights_out = Tensor(DataType::dt_float, {batch_size});
    instruction.context.pdf_set().call(
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
    tensor_copy(bias, output);
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

void backward_op_matmul(const Runtime::Instruction& instruction, TensorVec& locals) {
    // input, weight
    // output_grad
    // input_grad, weight_grad, bias_grad
    /*auto input = locals[instruction.input_indices[0]].contiguous();
    auto weight = locals[instruction.input_indices[1]].contiguous();
    auto output_grad = locals[instruction.input_indices[2]].contiguous();
    auto& input_grad = locals[instruction.output_indices[0]];
    auto& weight_grad = locals[instruction.output_indices[1]];
    auto& bias_grad = locals[instruction.output_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    input_grad = Tensor(DataType::dt_float, input.shape());
    weight_grad = Tensor(DataType::dt_float, weight.shape());
    bias_grad = Tensor(DataType::dt_float, {batch_size, dims_out});
    tensor_zero(input_grad);
    tensor_zero(weight_grad);
    tensor_zero(bias_grad);

    // compute grad_input = grad_output * weight
    {
        char transa = 'N', transb = 'N';
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

    // compute grad_weight = grad_output.T * input
    {
        char transa = 'T', transb = 'N';
        int m = dims_out, n = dims_in, k = batch_size;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = dims_out, ldc = batch_size;
        double* a = static_cast<double*>(input.data());
        double* b = static_cast<double*>(weight.data());
        double* c = static_cast<double*>(output.data());
        dgemm_(
            &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
        );
    }*/

    // compute grad_bias = sum_i grad_output_ij
}

}

void Runtime::initialize(
    const Function& function,
    Context& context,
    const std::vector<std::string>& grad_globals
) {
    locals_init.resize(function.locals.size());
    auto opt_function = optimize_constants(function);
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(opt_function);
    std::vector<bool> requires_grad(function.locals.size());

    for (auto& instr : opt_function.instructions) {
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        bool eval_grad = false;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
            eval_grad |= requires_grad.at(in.local_index);
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
            instr.instruction->opcode,
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            context,
            eval_grad
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({
                -1, {local_index}, {}, {}, {}, 0, context, requires_grad.at(local_index)
            });
        }
        ++instr_index;
    }

    for (auto& [name, value] : opt_function.globals) {
        Tensor global = context.global(name);
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
        if (std::find(
            grad_globals.begin(), grad_globals.end(), name
        ) != grad_globals.end()) {
            requires_grad.at(value.local_index) = true;
            grad_global_indices[name] = value.local_index;
        }
    }

    for (auto& local : opt_function.locals) {
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

    for (auto& out : opt_function.outputs) {
        output_indices.push_back(out.local_index);
    }
}

TensorVec Runtime::run(TensorVec& inputs) const {
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

std::tuple<TensorVec, TensorVec> Runtime::run_with_grad(TensorVec& inputs) const {
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        switch (instr.opcode) {
            case -1: // free memory
                if (!instr.eval_grad) {
                    locals[instr.input_indices[0]].reset();
                }
                break;
#include "runtime_mixin.h"
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    return {outputs, locals};
}

std::unordered_map<std::string, Tensor> Runtime::run_backward(
    TensorVec& output_grads, TensorVec& locals
) {
    TensorVec local_grads(locals.size());
    std::unordered_map<std::string, Tensor> global_grads;
    for (auto [index, grad] : std::views::zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (auto& instr : std::views::reverse(instructions)) {
        if (!instr.eval_grad) continue;
        switch (instr.opcode) {
#include "runtime_backward_mixin.h"
        }
    }
    for (auto& [name, index] : grad_global_indices) {
        global_grads[name] = locals[index];
    }

    return global_grads;
}

