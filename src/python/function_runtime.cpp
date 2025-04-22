#include "function_runtime.h"

#include <stdexcept>
#include <format>
#include <ranges>

using namespace madevent_py;

py::array_t<double> madevent_py::tensor_to_numpy(Tensor tensor) {
    auto data_raw = reinterpret_cast<double*>(tensor.data());
    py::capsule destroy(
        new Tensor(tensor),
        [](void* ptr) { delete static_cast<Tensor*>(ptr); }
    );
    SizeVec stride;
    std::size_t dtype_size = tensor.dtype_size();
    for (auto stride_item : tensor.stride()) {
        stride.push_back(dtype_size * stride_item);
    }
    return {tensor.shape(), stride, data_raw, destroy};
}

std::vector<py::array_t<double>> FunctionRuntime::call_numpy(std::vector<py::array> args) {
    // TODO: update numpy bindings
    auto n_args = function.inputs().size();
    if (args.size() != n_args) {
        throw std::invalid_argument(std::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    using Arr = py::array_t<double, py::array::f_style | py::array::forcecast>;
    std::vector<Arr> arrays;
    std::vector<Tensor> inputs;
    for (int i = 0; i < n_args; ++i) {
        auto arr = Arr::ensure(args.at(i));
        if (!arr) {
            throw std::invalid_argument(std::format("Argument {}: wrong dtype", i));
        }
        auto& input_type = function.inputs().at(i).type;
        if (arr.ndim() != input_type.shape.size() + 1) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                i, input_type.shape.size() + 1, arr.ndim()
            ));
        }
        std::vector<size_t> shape;
        for (int j = 0; j < arr.ndim(); ++j) {
            auto arr_size = arr.shape(j);
            if (j != 0 && arr_size != input_type.shape.at(j - 1)) {
                throw std::invalid_argument(std::format(
                    "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                    i, j, input_type.shape.at(j - 1), arr_size
                ));
            }
            shape.push_back(arr_size);
        }
        inputs.emplace_back(DataType::dt_float, shape, arr.mutable_data(), []{});
        arrays.push_back(arr);
    }

    if (!cpu_runtime) {
        if (context) {
            if (context->device() != cpu_device()) {
                throw std::invalid_argument("Given context does not have device CPU");
            }
            cpu_runtime = build_runtime(function, context);
        } else {
            cpu_runtime = build_runtime(function, madevent::default_context());
        }
    }
    return cpu_runtime->run(inputs)
        | std::views::transform(tensor_to_numpy)
        | std::ranges::to<std::vector<py::array_t<double>>>();
}
