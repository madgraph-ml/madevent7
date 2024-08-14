#include "madevent/backend/cpu/runtime.h"

#include <optional>
#include <cmath>
#include <tuple>
#include <array>
#include <functional>

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
    auto inputs = create_array<NIn>([&](auto i) -> Tensor& {
        auto& input = *locals[instruction.input_indices[i]];
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
    auto outputs = create_array<NOut>([&](auto i) -> Tensor& {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        std::vector<std::size_t> shape {batch_size};
        shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output.emplace(instruction.output_dtypes[i], shape);
        return *output;
    });

    auto views = std::apply([](auto&&... args){
        return std::make_tuple(args.view()...);
    }, std::tuple_cat(inputs, outputs));

    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        std::apply([i](auto&&... args) { function(args[i]...); }, views);
    }
}

}

Tensor::Tensor(DataType _dtype, SizeVec _shape) : dtype(_dtype), shape(_shape) {
    size_t stride_prod;
    switch (_dtype) {
        case DT_BOOL: stride_prod = sizeof(bool); break;
        case DT_INT: stride_prod = sizeof(int); break;
        case DT_FLOAT: stride_prod = sizeof(double); break;
    }
    for (auto size : _shape) {
        stride.push_back(stride_prod);
        stride_prod *= size;
    }
    data = std::make_shared<std::vector<uint8_t>>(stride_prod);
}

Runtime::Runtime(const Function& function) {

}

std::vector<Tensor> Runtime::run(std::vector<Tensor>& inputs) const {
    LocalVec locals(local_count);
    for (auto& instr : instructions) {
        switch (instr.opcode) {
            case -1: // free memory
                locals[instr.input_indices[0]].reset();
                break;
            /*case 34:
                batch_foreach<3, 2>(kernel_uniform_invariant_inverse, instr, locals);
                break;*/
#include "runtime_mixin.h"
        }
    }
    std::vector<Tensor> outputs;
    for (auto index : output_indices) {
        outputs.push_back(*locals[index]);
    }
    return outputs;
}
