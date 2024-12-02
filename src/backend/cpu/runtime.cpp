#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode/optimizer.h"
#include "kernels.h"

#include <optional>
#include <tuple>
#include <array>
#include <functional>
#include <algorithm>

using namespace madevent;
using namespace madevent::cpu;

namespace {

// call function(i) with argument i=0...N-1 and return the results as a tuple
template<std::size_t N, typename F, std::size_t... i>
constexpr auto range_to_tuple_impl(F&& function, std::index_sequence<i...>) {
    return std::make_tuple(function(i)...);
}
template<std::size_t N, typename F>
constexpr auto range_to_tuple(F&& function) {
    return range_to_tuple_impl<N>(std::forward<F>(function), std::make_index_sequence<N>{});
}

// return the tuple of TensorViews where the type is extracted from the signature of F
template<typename F, bool flatten> struct get_views;
template<typename... TParam, bool flatten>
struct get_views<void(*)(TParam...), flatten> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple(
            args.template view<typename TParam::DType, TParam::dim + 1>(flatten)...
        );
    }
};

template<auto function, int NIn, int NOut, bool flatten>
void batch_foreach(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    std::size_t batch_size = 1;
    auto inputs = range_to_tuple<NIn>([&](auto i) {
        auto input = locals[instruction.input_indices[i]];
        auto input_size = input.size(0);
        if (input_size != 1) {
            if (batch_size == 1) {
                batch_size = input_size;
            } else if (input_size != batch_size) {
                throw std::runtime_error("incompatible input shapes");
            }
        }
        return input;
    });
    auto outputs = range_to_tuple<NOut>([&](auto i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        SizeVec shape {batch_size};
        shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output = Tensor(instruction.output_dtypes[i], shape);
        return output;
    });

    // get views to the tensors with the correct types based on the signature of function
    auto views = std::apply(
        get_views<decltype(function), flatten>(), std::tuple_cat(inputs, outputs)
    );

    //#pragma omp parallel for
    for (std::size_t i = 0; i < batch_size; ++i) {
        std::apply([i](auto&&... args) { function(args[i]...); }, views);
    }
}

// Some helper definitions to use with std::visit and std::variant
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

void tensor_copy(Tensor source, Tensor target) {
    uint8_t* source_ptr = source.bytes();
    uint8_t* target_ptr = target.bytes();
    std::size_t contiguous_dims = std::min(source.contiguous_dims(), target.contiguous_dims());
    auto& shape = source.shape();
    std::size_t dim_count = shape.size();
    if (contiguous_dims == dim_count) {
        std::memcpy(target_ptr, source_ptr, source.stride().back() * shape.back());
        return;
    }

    std::size_t copy_size = contiguous_dims == 0 ?
        source.dtype_size() :
        source.stride()[contiguous_dims - 1] * shape[contiguous_dims - 1];
    SizeVec indices(dim_count - contiguous_dims);
    SizeVec source_step_sizes, target_step_sizes;
    source_step_sizes.reserve(indices.size());
    target_step_sizes.reserve(indices.size());
    std::size_t prev_source_total_size = 0;
    std::size_t prev_target_total_size = 0;
    for (std::size_t i = contiguous_dims; i < dim_count; ++i) {
        std::size_t source_step_size = source.stride()[i];
        std::size_t target_step_size = target.stride()[i];
        source_step_sizes.push_back(source_step_size - prev_source_total_size);
        target_step_sizes.push_back(target_step_size - prev_target_total_size);
        prev_source_total_size = source_step_size * shape[i];
        prev_target_total_size = target_step_size * shape[i];
    }

    std::size_t max_dim = dim_count - contiguous_dims - 1;
    std::size_t dim = max_dim;
    while (true) {
        auto& index = indices[dim];
        if (index < shape[contiguous_dims + dim]) {
            if (dim == 0) {
                std::memcpy(target_ptr, source_ptr, copy_size);
                source_ptr += source_step_sizes[dim];
                target_ptr += target_step_sizes[dim];
                ++index;
            } else {
                --dim;
            }
        } else {
            if (dim == max_dim) {
                break;
            } else {
                index = 0;
                ++dim;
                source_ptr += source_step_sizes[dim];
                target_ptr += target_step_sizes[dim];
                ++indices[dim];
            }
        }

    }
}

void op_stack(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    std::size_t batch_size, index = 0;
    Tensor output;
    bool first = true;
    for (auto input_index : instruction.input_indices) {
        auto& input = locals[input_index];
        auto input_size = input.size(0);
        if (first) {
            batch_size = input_size;
            auto shape = input.shape();
            shape.insert(shape.begin() + 1, instruction.input_indices.size());
            output = Tensor(instruction.output_dtypes.front(), shape);
            first = false;
        } else if (input_size != batch_size) {
            throw std::runtime_error("incompatible input shapes");
        }
        tensor_copy(input, output.select(1, index));
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
}

void op_batch_split(Runtime::Instruction instruction, std::vector<Tensor>& locals) {
    SizeVec sizes;
    auto tensors = locals[0].split(0, sizes);
    auto output_index = instruction.output_indices.begin();
    for (auto& tensor : tensors) {
        locals[*output_index] = tensor;
        ++output_index;
    }
}

}

Runtime::Runtime(const Function& function) : locals_init(function.locals.size()) {
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(function);
    for (auto& instr : function.instructions) {
        SizeVec input_indices;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
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
            output_shapes
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({-1, {local_index}, {}, {}, {}});
        }
        ++instr_index;
    }

    for (auto& local : function.locals) {
        std::visit(overloaded{
            [local, this](auto val) {
                Tensor tensor(local.type.dtype, {1});
                tensor.template view<decltype(val), 0>() = val;
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
