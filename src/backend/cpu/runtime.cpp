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

using namespace madevent;
using namespace madevent::cpu;

namespace {

// call function(i) with argument i=0...N-1 and return the results as an array
template<std::size_t N, typename F, std::size_t... i>
constexpr auto range_to_array_impl(F&& function, std::index_sequence<i...>) {
    return std::array<Tensor, N>{function(i)...};
}
template<std::size_t N, typename F>
constexpr auto range_to_array(F&& function) {
    return range_to_array_impl<N>(std::forward<F>(function), std::make_index_sequence<N>{});
}

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void batch_foreach(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto inputs = range_to_array<n_in>([&](auto i) {
        return locals[instruction.input_indices[i]];
    });
    auto outputs = range_to_array<n_out>([&](auto i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        SizeVec shape {batch_size};
        shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output = Tensor(instruction.output_dtypes[i], shape);
        return output;
    });

    if constexpr (dims == 0) {
        tensor_foreach_dynamic<scalar_func, vector_func, n_in, n_out>(inputs, outputs, batch_size);
    } else {
        tensor_foreach<scalar_func, vector_func, n_in, n_out, dims>(inputs, outputs, batch_size);
    }
}

void tensor_copy(Tensor source, Tensor target) {
    tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
        {source}, {target}, target.size(0)
    );
}

void op_stack(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
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

void op_unstack(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    auto tensors = locals[instruction.input_indices[0]].unstack(1);
    auto output_index = instruction.output_indices.begin();
    for (auto& tensor : tensors) {
        locals[*output_index] = tensor;
        ++output_index;
    }
}

void op_batch_cat(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
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

void op_batch_split(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto tensors = locals[instruction.input_indices[0]].split(0, sizes);
    auto output_index = instruction.output_indices.begin();
    for (auto [tensor, output_index] : std::views::zip(tensors, instruction.output_indices)) {
        locals[output_index] = tensor;
    }
}

}

Runtime::Runtime(const Function& function) : locals_init(function.locals.size()) {
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(function);
    for (auto& instr : function.instructions) {
        SizeVec input_indices;
        std::size_t batch_size_index = 0;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one || in.type.shape.size() > 0) {
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
            instr.instruction->opcode,
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({-1, {local_index}, {}, {}, {}});
        }
        ++instr_index;
    }

    for (auto& local : function.locals) {
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
                    auto tview = tensor.template view<typename decltype(items)::value_type, 2>()[0];
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

    for (auto& out : function.outputs) {
        output_indices.push_back(out.local_index);
    }
}

std::vector<Tensor> Runtime::run(std::vector<Tensor>& inputs) const {
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
    std::vector<Tensor> outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    return outputs;
}
