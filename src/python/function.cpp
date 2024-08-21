#include "function.h"

#include <stdexcept>
#include <fmt/core.h>

using namespace madevent_py;

std::vector<py::array_t<double>> FunctionRuntime::call_numpy(std::vector<py::array> args) {
    auto n_args = function.inputs.size();
    if (args.size() != n_args) {
        throw std::invalid_argument(fmt::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    using Arr = py::array_t<double, py::array::f_style | py::array::forcecast>;
    std::vector<Arr> arrays;
    std::vector<cpu::Tensor> inputs;
    for (int i = 0; i < n_args; ++i) {
        auto arr = Arr::ensure(args[i]);
        if (!arr) {
            throw std::invalid_argument(fmt::format("Argument {}: wrong dtype", i));
        }
        auto& input_type = function.inputs[i].type;
        if (arr.ndim() != input_type.shape.size() + 1) {
            throw std::invalid_argument(fmt::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                i, input_type.shape.size() + 1, arr.ndim()
            ));
        }
        std::vector<size_t> shape;
        for (int j = 0; j < arr.ndim(); ++j) {
            auto arr_size = arr.shape(j);
            if (j != 0 && arr_size != input_type.shape[j-1]) {
                throw std::invalid_argument(fmt::format(
                    "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                    i, j, input_type.shape[j-1], arr_size
                ));
            }
            shape.push_back(arr_size);
        }
        inputs.emplace_back(DT_FLOAT, shape, std::shared_ptr<uint8_t[]>(
            reinterpret_cast<uint8_t*>(arr.mutable_data()), [](auto ptr) {}
        ));
        arrays.push_back(arr);
    }

    if (!cpu_runtime) {
        cpu_runtime.emplace(function);
    }
    auto outputs = cpu_runtime->run(inputs);
    std::vector<py::array_t<double>> outputs_numpy;
    for (auto& output : outputs) {
        auto data_raw = reinterpret_cast<double*>(output.data.get());
        py::capsule destroy(
            new cpu::Tensor::DataPtr(output.data),
            [](void* ptr) { delete static_cast<cpu::Tensor::DataPtr*>(ptr); }
        );
        outputs_numpy.emplace_back(output.shape, output.stride, data_raw, destroy);
    }
    return outputs_numpy;
}

#ifdef TORCH_FOUND
std::vector<torch::Tensor> FunctionRuntime::call_torch(std::vector<torch::Tensor> args) {
    auto n_args = function.inputs.size();
    if (args.size() != n_args) {
        throw std::invalid_argument(fmt::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    std::vector<torch::Tensor> tensors;
    std::vector<cpu::Tensor> inputs;
    for (int i = 0; i < n_args; ++i) {
        auto n_dims = args[i].dim();
        std::vector<long int> permutation;
        for (int k = n_dims-1; k >= 0; --k) {
            permutation.push_back(k);
        }
        auto tensor = args[i]
            .permute(permutation)
            .to(torch::kFloat64)
            .contiguous()
            .permute(permutation);
        auto& input_type = function.inputs[i].type;
        if (n_dims != input_type.shape.size() + 1) {
            throw std::invalid_argument(fmt::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                i, input_type.shape.size() + 1, n_dims
            ));
        }
        std::vector<size_t> shape;
        for (int j = 0; j < n_dims; ++j) {
            auto tensor_size = tensor.size(j);
            if (j != 0 && tensor_size != input_type.shape[j-1]) {
                throw std::invalid_argument(fmt::format(
                    "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                    i, j, input_type.shape[j-1], tensor_size
                ));
            }
            shape.push_back(tensor_size);
        }
        inputs.emplace_back(DT_FLOAT, shape, std::shared_ptr<uint8_t[]>(
            reinterpret_cast<uint8_t*>(tensor.data_ptr<double>()), [](auto ptr) {}
        ));
        tensors.push_back(tensor);
    }

    if (!cpu_runtime) {
        cpu_runtime.emplace(function);
    }
    auto outputs = cpu_runtime->run(inputs);
    std::vector<torch::Tensor> output_tensors;
    for (auto& output : outputs) {
        auto ptr = new cpu::Tensor::DataPtr(output.data);
        std::vector<long int> shape {output.shape.begin(), output.shape.end()};
        std::vector<long int> stride;
        for (auto s : output.stride) {
            stride.push_back(s / sizeof(double));
        }
        output_tensors.push_back(torch::from_blob(
            output.data.get(),
            shape,
            stride,
            [&ptr] (void* data) { delete ptr; },
            torch::TensorOptions().dtype(torch::kFloat64)
        ));

        /*auto data_raw = reinterpret_cast<double*>(output.data.get());
        py::capsule destroy(
            new cpu::Tensor::DataPtr(output.data),
            [](void* ptr) { delete static_cast<cpu::Tensor::DataPtr*>(ptr); }
        );
        outputs_numpy.emplace_back(output.shape, output.stride, data_raw, destroy);*/
    }
    return output_tensors;
}
#endif
