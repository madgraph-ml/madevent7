#include "function.h"

#include <stdexcept>
#include <format>

using namespace madevent_py;

std::vector<py::array_t<double>> FunctionRuntime::call_numpy(std::vector<py::array> args) {
    // TODO: update numpy bindings
    auto n_args = function.inputs.size();
    if (args.size() != n_args) {
        throw std::invalid_argument(std::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    using Arr = py::array_t<double, py::array::f_style | py::array::forcecast>;
    std::vector<Arr> arrays;
    std::vector<Tensor> inputs;
    for (int i = 0; i < n_args; ++i) {
        auto arr = Arr::ensure(args[i]);
        if (!arr) {
            throw std::invalid_argument(std::format("Argument {}: wrong dtype", i));
        }
        auto& input_type = function.inputs[i].type;
        if (arr.ndim() != input_type.shape.size() + 1) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                i, input_type.shape.size() + 1, arr.ndim()
            ));
        }
        std::vector<size_t> shape;
        for (int j = 0; j < arr.ndim(); ++j) {
            auto arr_size = arr.shape(j);
            if (j != 0 && arr_size != input_type.shape[j-1]) {
                throw std::invalid_argument(std::format(
                    "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                    i, j, input_type.shape[j-1], arr_size
                ));
            }
            shape.push_back(arr_size);
        }
        inputs.emplace_back(DataType::dt_float, shape, arr.mutable_data());
        arrays.push_back(arr);
    }

    if (!cpu_runtime) {
        if (context) {
            if (context->device() != cpu_device()) {
                throw std::invalid_argument("Given context does not have device CPU");
            }
            cpu_runtime.emplace(function, context);
        } else {
            cpu_runtime.emplace(function);
        }
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
    //TODO: check batch sizes
    auto n_args = function.inputs.size();
    if (args.size() != n_args) {
        throw std::invalid_argument(std::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    std::vector<torch::Tensor> tensors;
    std::vector<Tensor> inputs;
    bool is_cuda = false;
    DevicePtr device = cpu_device();
    for (int i = 0; i < n_args; ++i) {
        auto& arg = args.at(i);
        auto n_dims = arg.dim();
        std::vector<int64_t> permutation;
        for (int k = n_dims-1; k >= 0; --k) {
            permutation.push_back(k);
        }
        auto arg_dtype = arg.scalar_type();
        auto& input_type = function.inputs.at(i).type;
        bool is_batch_sizes = input_type.dtype == DataType::batch_sizes;
        torch::Dtype target_dtype;
        bool dtype_ok = false;
        if (arg_dtype == torch::kFloat64 || arg_dtype == torch::kFloat32) {
            target_dtype = torch::kFloat64;
            dtype_ok = input_type.dtype == DataType::dt_float;
        } else if (arg_dtype == torch::kInt64 || arg_dtype == torch::kInt32) {
            target_dtype = torch::kInt64;
            dtype_ok = input_type.dtype == DataType::dt_int || is_batch_sizes;
        } else if (arg_dtype == torch::kBool) {
            target_dtype = torch::kBool;
            dtype_ok = input_type.dtype == DataType::dt_bool;
        }
        if (!dtype_ok) {
            throw std::invalid_argument(std::format("Argument {}: dtype not accepted", i));
        }
        auto tensor = arg
            .permute(permutation)
            .to(target_dtype)
            .contiguous()
            .permute(permutation);
        tensors.push_back(tensor); // Important: prevents tensor from getting destroyed
        if (i == 0) {
            is_cuda = tensor.is_cuda();
            if (is_cuda) {
#ifdef CUDA_FOUND
                device = cuda_device();
#else
                throw std::runtime_error("madevent was compiled without cuda support");
#endif
            }
        } else if (!is_batch_sizes && is_cuda != tensor.is_cuda()) {
            throw std::invalid_argument("All inputs have to be on the same device.");
        }
        if (is_batch_sizes) {
            if (n_dims != 1) {
                throw std::invalid_argument(std::format(
                    "Argument {}: wrong input dimension. Expected 1, got {}", i, n_dims
                ));
            }
            if (tensor.size(0) != input_type.batch_size_list.size()) {
                throw std::invalid_argument(std::format(
                    "Argument {}, dimension 0: shape mismatch. Expected {}, got {}",
                    i, input_type.batch_size_list.size(), tensor.size(0)
                ));
            }
            std::vector<std::size_t> batch_sizes(
                tensor.data_ptr<long long>(), tensor.data_ptr<long long>() + tensor.numel()
            );
            inputs.emplace_back(batch_sizes);
        } else if (input_type.batch_size == BatchSize::one && input_type.shape.size() == 0) {
            switch(input_type.dtype) {
            case DataType::dt_float:
                inputs.emplace_back(tensor.item<double>(), device);
                break;
            case DataType::dt_int:
                inputs.emplace_back(tensor.item<long long>(), device);
                break;
            case DataType::dt_bool:
                inputs.emplace_back(tensor.item<bool>(), device);
                break;
            default:
                break;
            }
        } else {
            bool is_batch = input_type.batch_size != BatchSize::one;
            if (n_dims != input_type.shape.size() + is_batch) {
                throw std::invalid_argument(std::format(
                    "Argument {}: wrong input dimension. Expected {}, got {}",
                    i, input_type.shape.size() + 1, n_dims
                ));
            }
            std::vector<size_t> shape;
            if (!is_batch) {
                shape.push_back(1);
            }
            for (int j = 0; j < n_dims; ++j) {
                auto tensor_size = tensor.size(j);
                if (j != 0 && tensor_size != input_type.shape.at(j - is_batch)) {
                    throw std::invalid_argument(std::format(
                        "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                        i, j, input_type.shape.at(j - is_batch), tensor_size
                    ));
                }
                shape.push_back(tensor_size);
            }
            void* data_ptr;
            if (target_dtype == torch::kFloat64) {
                data_ptr = tensor.data_ptr<double>();
            } else if (target_dtype == torch::kInt64) {
                data_ptr = tensor.data_ptr<int64_t>();
            } else {
                data_ptr = tensor.data_ptr<bool>();
            }
            inputs.emplace_back(input_type.dtype, shape, device, data_ptr);
        }
    }

    std::vector<Tensor> outputs;
    if (is_cuda) {
#ifdef CUDA_FOUND
        //TODO: update for context
        if (!cuda_runtime) {
            cuda_runtime.emplace(function);
        }
        outputs = cuda_runtime->run(inputs);
#endif
    } else {
        if (!cpu_runtime) {
            if (context) {
                if (context->device() != cpu_device()) {
                    throw std::invalid_argument("Given context does not have device CPU");
                }
                cpu_runtime.emplace(function, context);
            } else {
                cpu_runtime.emplace(function);
            }
        }
        outputs = cpu_runtime->run(inputs);
    }
    std::vector<torch::Tensor> output_tensors;
    for (auto& output : outputs) {
        std::vector<int64_t> shape {output.shape().begin(), output.shape().end()};
        std::vector<int64_t> stride;
        auto dtype_size = output.dtype_size();
        for (auto s : output.stride()) {
            stride.push_back(s / dtype_size);
        }
        if (output.dtype() == DataType::batch_sizes) {
            auto& batch_sizes = output.batch_sizes();
            torch::Tensor tensor = torch::zeros(
                {static_cast<int64_t>(batch_sizes.size())},
                torch::TensorOptions().dtype(torch::kInt64)
            );
            auto accessor = tensor.accessor<long long, 1>();
            std::size_t i = 0;
            for (auto size : batch_sizes) {
                accessor[i] = size;
                ++i;
            }
            output_tensors.push_back(tensor);
        } else {
            torch::Dtype dtype;
            switch(output.dtype()) {
                case DataType::dt_float: dtype = torch::kFloat64; break;
                case DataType::dt_int: dtype = torch::kInt64; break;
                case DataType::dt_bool: dtype = torch::kBool; break;
                default: break;
            }
            output_tensors.push_back(torch::from_blob(
                output.data(),
                shape,
                stride,
                [output] (void* data) mutable { output.reset(); },
                torch::TensorOptions()
                    .dtype(dtype)
                    .device(is_cuda ? torch::kCUDA : torch::kCPU)
            ));
        }
    }
    return output_tensors;
}
#endif
