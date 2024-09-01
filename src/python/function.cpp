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
    std::vector<Tensor> inputs;
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
        inputs.emplace_back(DT_FLOAT, shape, arr.mutable_data());
        arrays.push_back(arr);
    }

    if (!cpu_runtime) {
        cpu_runtime.emplace(function);
    }
    auto outputs = cpu_runtime->run(inputs);
    std::vector<py::array_t<double>> outputs_numpy;
    for (auto& output : outputs) {
        auto data_raw = reinterpret_cast<double*>(output.data());
        py::capsule destroy(
            new Tensor(output),
            [](void* ptr) { delete static_cast<Tensor*>(ptr); }
        );
        outputs_numpy.emplace_back(output.shape(), output.stride(), data_raw, destroy);
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
    std::vector<Tensor> inputs;
    bool is_cuda = false;
    Device& device = cpu_device();
    for (int i = 0; i < n_args; ++i) {
        auto n_dims = args[i].dim();
        std::vector<int64_t> permutation;
        for (int k = n_dims-1; k >= 0; --k) {
            permutation.push_back(k);
        }
        auto tensor = args[i]
            .permute(permutation)
            .to(torch::kFloat64)
            .contiguous()
            .permute(permutation);
        if (i == 0) {
            is_cuda = tensor.is_cuda();
            if (is_cuda) {
#ifdef CUDA_FOUND
                device = cuda_device();
#else
                throw std::runtime_error("madevent was compiled without cuda support");
#endif
            }
        } else if (is_cuda != tensor.is_cuda()) {
            throw std::invalid_argument("All inputs have to be on the same device.");
        }
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
        inputs.emplace_back(DT_FLOAT, shape, device, tensor.data_ptr<double>());
        tensors.push_back(tensor);
    }

    std::vector<Tensor> outputs;
    if (is_cuda) {
#ifdef CUDA_FOUND
        if (!cuda_runtime) {
            cuda_runtime.emplace(function);
        }
        outputs = cuda_runtime->run(inputs);
#endif
    } else {
        if (!cpu_runtime) {
            cpu_runtime.emplace(function);
        }
        outputs = cpu_runtime->run(inputs);
    }
    std::vector<torch::Tensor> output_tensors;
    for (auto& output : outputs) {
        std::vector<int64_t> shape {output.shape().begin(), output.shape().end()};
        std::vector<int64_t> stride;
        for (auto s : output.stride()) {
            stride.push_back(s / sizeof(double));
        }
        output_tensors.push_back(torch::from_blob(
            output.data(),
            shape,
            stride,
            [output] (void* data) mutable { output.reset(); },
            torch::TensorOptions()
                .dtype(torch::kFloat64)
                .device(is_cuda ? torch::kCUDA : torch::kCPU)
        ));
    }
    return output_tensors;
}
#endif
