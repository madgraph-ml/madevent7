#include "madevent/backend/cpu/runtime.h"

#include <optional>
#include <cmath>
#include <tuple>
#include <array>
#include <functional>
#include <algorithm>

#define KERNELSPEC

using namespace madevent::cpu;

namespace {

using std::sqrt;
using std::sin;
using std::cos;
using std::sinh;
using std::cosh;
using std::atan2;
using std::pow;
using std::fabs;
using std::log;
using std::tan;
using std::atan;
using std::exp;

using DoubleInput = const TensorView<double>;
using DoubleOutput = TensorView<double>;

#include "../common/kernels.h"

using LocalVec = std::vector<std::optional<Tensor>>;

template<std::size_t N, typename F, std::size_t... I>
constexpr auto create_array_impl(F&& function, std::index_sequence<I...>) {
    return std::make_tuple(function(I)...);
}

template<std::size_t N, typename F>
constexpr auto create_array(F&& function) {
    return create_array_impl<N>(std::forward<F>(function), std::make_index_sequence<N>{});
}

template<auto function, int NIn, int NOut>
void batch_foreach(Runtime::Instruction instruction, LocalVec& locals) {
    std::size_t batch_size = 1;
    auto inputs = create_array<NIn>([&](auto i) {
        auto& input = *locals[instruction.input_indices[i]];
        auto input_size = input.size(0);
        if (input_size != 1) {
            if (batch_size == 1) {
                batch_size = input_size;
            } else if (input_size != batch_size) {
                throw std::runtime_error("incompatible input shapes");
            }
        }
        for (int j = 0; j < batch_size; j++) {
            double d = static_cast<DoubleInput>(input.view()[j]);
            std::cout << i << " " << j << " " << d << "\n";
        }
        return input.view();
    });
    auto outputs = create_array<NOut>([&](auto i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        SizeVec shape {batch_size};
        shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output.emplace(instruction.output_dtypes[i], shape);
        return output->view();
    });

    //#pragma omp parallel for
    //#pragma omp for simd
    for (std::size_t i = 0; i < batch_size; ++i) {
        std::apply(
            [i](auto&&... args) { function(args[i]...); },
            std::tuple_cat(inputs, outputs)
        );
    }
}

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

}

Tensor::Tensor(DataType _dtype, SizeVec _shape, DataPtr _data)
    : dtype(_dtype), shape(_shape), data(_data)
{
    std::size_t stride_prod;
    switch (_dtype) {
        case DT_BOOL: stride_prod = sizeof(bool); break;
        case DT_INT: stride_prod = sizeof(int); break;
        case DT_FLOAT: stride_prod = sizeof(double); break;
    }
    bool first = true;
    for (auto size : _shape) {
        if (first && size == 1) {
            stride.push_back(0);
        } else {
            stride.push_back(stride_prod);
        }
        first = false;
        stride_prod *= size;
    }
    if (!data) {
        data = std::shared_ptr<uint8_t[]>(new uint8_t[stride_prod]);
    }
}

Runtime::Runtime(const Function& function) {
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
            instr.instruction->opcode, input_indices, output_indices, output_dtypes, output_shapes
        });
    }

    for (auto& local : function.locals) {
        if (local.literal_value == std::monostate{}) {
            continue;
        }
        std::visit(overloaded{
            [local, &locals_init](auto val) {
                Tensor tensor(local.dtype, {1});
                tensor.view<decltype(val)>() = val;
                locals_init[local.local_index] = tensor;
            },
            [](std::string val){}
            [](std::monostate val){}
        }, local.literal_value); 
    }

    for (auto& out : function.outputs) {
        output_indices.push_back(out.local_index);
    }
}

std::vector<Tensor> Runtime::run(std::vector<Tensor>& inputs) const {
    LocalVec locals(locals_init);
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
        outputs.push_back(*locals[index]);
    }
    return outputs;
}
