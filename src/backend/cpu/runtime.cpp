#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode/optimizer.h"
#include "madevent/util.h"
#include "kernels.h"

#include <optional>
#include <tuple>
#include <array>
#include <functional>
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

// call function(i) with argument i=0...N-1 and return the results as an array
template<std::size_t N, typename F, std::size_t... i>
constexpr auto range_to_array_impl(F&& function, std::index_sequence<i...>) {
    return std::array<Tensor*, N>{function(i)...};
}
template<std::size_t N, typename F>
constexpr auto range_to_array(F&& function) {
    return range_to_array_impl<N>(
        std::forward<F>(function), std::make_index_sequence<N>{}
    );
}

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void batch_foreach(const Runtime::Instruction& instruction, TensorVec& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto inputs = range_to_array<n_in>([&](auto i) {
        return &locals[instruction.input_indices[i]];
    });
    auto outputs = range_to_array<n_out>([&](auto i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        SizeVec shape(output_shape.size() + 1);
        shape[0] = batch_size;
        std::copy(output_shape.begin(), output_shape.end(), shape.begin() + 1);
        //SizeVec shape {batch_size};
        //shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output = Tensor(instruction.output_dtypes[i], shape);
        return &output;
    });

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

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void backward_batch_foreach(
    const Runtime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto inputs = range_to_array<n_in>([&](auto i) {
        return locals[instruction.input_indices[i]];
    });
    auto output_grads = range_to_array<n_out>([&](auto i) {
        return local_grads[instruction.output_indices[i]];
    });
    auto input_grads = range_to_array<n_out>([&](auto i) {
        auto& input_grad = locals[instruction.input_indices[i]];
        if (!input_grad) {
            auto& grad_shape = inputs[i].shape();
            SizeVec shape {batch_size};
            shape.insert(shape.end(), grad_shape.begin(), grad_shape.end());
            input_grad = Tensor(inputs[i].dtype(), shape);
            //TODO: zero init
        }
        return input_grad;
    });

    if constexpr (dims == 0) {
        tensor_foreach_dynamic<scalar_func, vector_func, n_in, n_out>(
            std::tuple_cat(inputs, output_grads), input_grads, batch_size
        );
    } else {
        tensor_foreach<scalar_func, vector_func, n_in, n_out, dims>(
            std::tuple_cat(inputs, output_grads), input_grads, batch_size
        );
    }
}

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

void op_stack(const Runtime::Instruction& instruction, TensorVec& locals) {
    auto shape = locals[instruction.input_indices[0]].shape();
    shape[0] = locals[instruction.batch_size_index].size(0);
    shape.insert(shape.begin() + 1, instruction.input_indices.size());
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

void Runtime::initialize(const Function& function, Context& context) {
    locals_init.resize(function.locals.size());
    auto opt_function = optimize_constants(function);
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(opt_function);
    std::vector<bool> requires_grad(function.locals.size());

    //TODO: initialize globals

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
        SizeVec full_shape {1};
        full_shape.insert(
            full_shape.begin(), value.type.shape.begin(), value.type.shape.end()
        );
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            throw std::invalid_argument(std::format(
                "Global {} has wrong dtype or shape", name
            ));
        }
        locals_init[value.local_index] = global;
    }

    for (auto& local : opt_function.locals) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, cpu_device());
                locals_init[local.local_index] = tensor;
            },
            [&](TensorValue val) {
                auto& [shape, items] = val;
                SizeVec full_shape{1};
                for (auto size : shape) {
                    full_shape.push_back(size);
                }
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

void Runtime::run_backward(TensorVec& output_grads, TensorVec& locals) {
    TensorVec local_grads(locals.size());
    for (auto [index, grad] : std::views::zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (auto& instr : std::views::reverse(instructions)) {
        switch (instr.opcode) {
//#include "runtime_backward_mixin.h"
        }
    }
}

