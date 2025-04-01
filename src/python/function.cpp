#include "function.h"

#include <stdexcept>
#include <format>
#include <ranges>

using namespace madevent_py;

namespace {

template<
    typename CpuFunc
#ifdef CUDA_FOUND
    , typename CudaFunc
#endif
>
std::vector<torch::Tensor> call_torch_impl(
    const std::vector<torch::Tensor>& args,
    const Function& function,
    ContextPtr context,
    std::optional<cpu::Runtime>& cpu_runtime,
    CpuFunc cpu_func
#ifdef CUDA_FOUND
    , std::optional<cuda::Runtime>& cuda_runtime
    , CudaFunc cuda_func
#endif
) {
    //TODO: check batch sizes
    auto n_args = function.inputs().size();
    if (args.size() != n_args) {
        throw std::invalid_argument(std::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    std::vector<Tensor> inputs;
    bool is_cuda = false;
    DevicePtr expected_device = nullptr;
    for (int i = 0; i < n_args; ++i) {
        auto& arg = args.at(i);
        auto& input_type = function.inputs().at(i).type;
        auto tensor = torch_to_tensor(arg, input_type, i, expected_device);
        if (i == 0) expected_device = tensor.device();
        inputs.push_back(tensor);
    }

    std::vector<Tensor> outputs;
    if (is_cuda) {
#ifdef CUDA_FOUND
        //TODO: update for context
        if (!cuda_runtime) {
            cuda_runtime.emplace(function);
        }
        outputs = cuda_func(inputs);
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
        outputs = cpu_func(inputs);
    }
    return outputs
        | std::views::transform(tensor_to_torch)
        | std::ranges::to<std::vector<torch::Tensor>>();
}

}
py::array_t<double> madevent_py::tensor_to_numpy(Tensor tensor) {
    auto data_raw = reinterpret_cast<double*>(tensor.data());
    py::capsule destroy(
        new Tensor(tensor),
        [](void* ptr) { delete static_cast<Tensor*>(ptr); }
    );
    return {tensor.shape(), tensor.stride(), data_raw, destroy};
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
            cpu_runtime.emplace(function, context);
        } else {
            cpu_runtime.emplace(function);
        }
    }
    return cpu_runtime->run(inputs)
        | std::views::transform(tensor_to_numpy)
        | std::ranges::to<std::vector<py::array_t<double>>>();
}

#ifdef TORCH_FOUND

torch::Tensor madevent_py::tensor_to_torch(Tensor tensor) {
    std::vector<int64_t> shape {tensor.shape().begin(), tensor.shape().end()};
    std::vector<int64_t> stride;
    auto dtype_size = tensor.dtype_size();
    for (auto s : tensor.stride()) {
        stride.push_back(s / dtype_size);
    }
    if (tensor.dtype() == DataType::batch_sizes) {
        auto& batch_sizes = tensor.batch_sizes();
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
        return tensor;
    } else {
        torch::Dtype dtype;
        switch(tensor.dtype()) {
            case DataType::dt_float: dtype = torch::kFloat64; break;
            case DataType::dt_int: dtype = torch::kInt64; break;
            case DataType::dt_bool: dtype = torch::kBool; break;
            default: break;
        }
        return torch::from_blob(
            tensor.data(),
            shape,
            stride,
            [tensor] (void* data) mutable { tensor.reset(); },
            torch::TensorOptions()
                .dtype(dtype)
                .device(tensor.device() == cpu_device() ? torch::kCPU : torch::kCUDA)
        );
    }
}

std::optional<torch::Tensor> madevent_py::tensor_to_torch_opt(Tensor tensor) {
    if (tensor) {
        return tensor_to_torch(tensor);
    } else {
        return std::nullopt;
    }
}

Tensor madevent_py::torch_to_tensor(
    torch::Tensor torch_tensor,
    Type expected_type,
    std::size_t arg_index,
    DevicePtr expected_device
) {
    auto n_dims = torch_tensor.dim();
    std::vector<int64_t> permutation;
    for (int k = n_dims-1; k >= 0; --k) {
        permutation.push_back(k);
    }
    auto arg_dtype = torch_tensor.scalar_type();
    bool is_batch_sizes = expected_type.dtype == DataType::batch_sizes;
    torch::Dtype target_dtype;
    bool dtype_ok = false;
    if (arg_dtype == torch::kFloat64 || arg_dtype == torch::kFloat32) {
        target_dtype = torch::kFloat64;
        dtype_ok = expected_type.dtype == DataType::dt_float;
    } else if (arg_dtype == torch::kInt64 || arg_dtype == torch::kInt32) {
        target_dtype = torch::kInt64;
        dtype_ok = expected_type.dtype == DataType::dt_int || is_batch_sizes;
    } else if (arg_dtype == torch::kBool) {
        target_dtype = torch::kBool;
        dtype_ok = expected_type.dtype == DataType::dt_bool;
    }
    if (!dtype_ok) {
        throw std::invalid_argument(std::format("Argument {}: dtype not accepted", arg_index + 1));
    }
    auto tensor = torch_tensor
        .permute(permutation)
        .to(target_dtype)
        .contiguous()
        .permute(permutation);

    DevicePtr device;
    if (tensor.is_cuda()) {
#ifdef CUDA_FOUND
        device = cuda_device();
#else
        throw std::runtime_error("madevent was compiled without cuda support");
#endif
    } else {
        device = cpu_device();
    }
    if (expected_device && device != expected_device) {
        throw std::invalid_argument("All inputs have to be on the same device.");
    }
    if (is_batch_sizes) {
        if (n_dims != 1) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected 1, got {}", arg_index + 1, n_dims
            ));
        }
        if (tensor.size(0) != expected_type.batch_size_list.size()) {
            throw std::invalid_argument(std::format(
                "Argument {}, dimension 0: shape mismatch. Expected {}, got {}",
                arg_index + 1, expected_type.batch_size_list.size(), tensor.size(0)
            ));
        }
        std::vector<std::size_t> batch_sizes(
            tensor.data_ptr<long long>(), tensor.data_ptr<long long>() + tensor.numel()
        );
        return {batch_sizes};
    } else if (expected_type.batch_size == BatchSize::one && expected_type.shape.size() == 0) {
        switch(expected_type.dtype) {
        case DataType::dt_float:
            return {tensor.item<double>(), device};
        case DataType::dt_int:
            return {tensor.item<long long>(), device};
        case DataType::dt_bool:
            return {tensor.item<bool>(), device};
        default:
            std::unreachable();
        }
    } else {
        bool is_batch = expected_type.batch_size != BatchSize::one;
        if (n_dims != expected_type.shape.size() + is_batch) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                arg_index + 1, expected_type.shape.size() + 1, n_dims
            ));
        }
        std::vector<size_t> shape;
        if (!is_batch) {
            shape.push_back(1);
        }
        for (int j = 0; j < n_dims; ++j) {
            auto tensor_size = tensor.size(j);
            if (j != 0 && tensor_size != expected_type.shape.at(j - is_batch)) {
                throw std::invalid_argument(std::format(
                    "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                    arg_index + 1, j, expected_type.shape.at(j - is_batch), tensor_size
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
        return {
            expected_type.dtype, shape, device, data_ptr,
            [tensor] mutable { tensor = torch::Tensor(); }
        };
    }
}

Tensor madevent_py::torch_to_tensor_unchecked(std::optional<torch::Tensor> torch_tensor) {
    if (!torch_tensor) return {};

    auto n_dims = torch_tensor->dim();
    std::vector<int64_t> permutation;
    for (int k = n_dims-1; k >= 0; --k) {
        permutation.push_back(k);
    }
    auto torch_dtype = torch_tensor->scalar_type();
    torch::Dtype target_dtype;
    DataType dtype;
    if (torch_dtype == torch::kFloat64 || torch_dtype == torch::kFloat32) {
        target_dtype = torch::kFloat64;
        dtype = DataType::dt_float;
    } else if (torch_dtype == torch::kInt64 || torch_dtype == torch::kInt32) {
        target_dtype = torch::kInt64;
        dtype = DataType::dt_int;
    } else if (torch_dtype == torch::kBool) {
        target_dtype = torch::kBool;
        dtype = DataType::dt_bool;
    } else {
        throw std::invalid_argument("dtype not accepted");
    }
    auto tensor = torch_tensor
        ->permute(permutation)
        .to(target_dtype)
        .contiguous()
        .permute(permutation);

    void* data_ptr;
    if (target_dtype == torch::kFloat64) {
        data_ptr = torch_tensor->data_ptr<double>();
    } else if (target_dtype == torch::kInt64) {
        data_ptr = torch_tensor->data_ptr<int64_t>();
    } else {
        data_ptr = torch_tensor->data_ptr<bool>();
    }
    SizeVec shape;
    for (std::size_t i = 0; i < n_dims; ++i) {
        shape.push_back(tensor.size(i));
    }
    return {
        dtype, shape, cpu_device(), data_ptr,
        [tensor] mutable { tensor = torch::Tensor(); }
    };
}

std::vector<torch::Tensor> FunctionRuntime::call_torch(
    const std::vector<torch::Tensor>& args
) {
    return call_torch_impl(
        args,
        function,
        context,
        cpu_runtime,
        [&] (auto inputs) { return cpu_runtime->run(inputs); }
#ifdef CUDA_FOUND
        , cuda_runtime
        , [&] (auto inputs) { return cuda_runtime->run(inputs); }
#endif
    );
}

std::tuple<
    std::vector<torch::Tensor>,
    std::vector<std::optional<torch::Tensor>>,
    std::vector<bool>
> FunctionRuntime::call_with_grad_torch(
    const std::vector<torch::Tensor>& args,
    const std::vector<bool>& input_requires_grad
) {
    std::vector<std::optional<torch::Tensor>> local_grads;
    std::vector<bool> eval_grad;
    auto outputs = call_torch_impl(
        args,
        function,
        context,
        cpu_runtime,
        [&] (auto inputs) {
            auto [out, loc_grad, ev_grad] = cpu_runtime->run_with_grad(
                inputs, input_requires_grad
            );
            local_grads = loc_grad
                | std::views::transform(tensor_to_torch_opt)
                | std::ranges::to<std::vector<std::optional<torch::Tensor>>>();
            eval_grad = ev_grad;
            return out;
        }
#ifdef CUDA_FOUND
        , cuda_runtime
        , [&] (auto inputs) {
            auto [out, loc_grad, ev_grad] = cuda_runtime->run_with_grad(
                inputs, input_requires_grad
            );
            local_grads = loc_grad
                | std::views::transform(tensor_to_torch_opt)
                | std::ranges::to<std::vector<std::optional<torch::Tensor>>>();
            eval_grad = ev_grad;
            return out;
        }
#endif
    );
    return {outputs, local_grads, eval_grad};
}
std::tuple<
    std::vector<std::optional<torch::Tensor>>,
    std::vector<std::tuple<std::string, std::optional<torch::Tensor>>>
> FunctionRuntime::call_backward_torch(
    const std::vector<torch::Tensor>& output_grads,
    const std::vector<std::optional<torch::Tensor>>& stored_locals,
    const std::vector<bool>& eval_grad
) {
    auto arg_out = output_grads
        | std::views::transform(torch_to_tensor_unchecked)
        | std::ranges::to<std::vector<Tensor>>();
    auto arg_locals = stored_locals
        | std::views::transform(torch_to_tensor_unchecked)
        | std::ranges::to<std::vector<Tensor>>();
    // TODO: checks here
    auto [ret_in_grads, ret_glob_grads] = cpu_runtime->run_backward(
        arg_out, arg_locals, eval_grad
    );
    auto input_grads = ret_in_grads
        | std::views::transform(tensor_to_torch_opt)
        | std::ranges::to<std::vector<std::optional<torch::Tensor>>>();
    using STP = std::tuple<std::string, std::optional<torch::Tensor>>;
    auto global_grads = ret_glob_grads
        | std::views::transform([] (auto& item) -> STP {
            return {std::get<0>(item), tensor_to_torch_opt(std::get<1>(item))};
        })
        | std::ranges::to<std::vector<STP>>();
    return {input_grads, global_grads};
}

#endif
