#include "torch.h"

#include <stdexcept>
#include <format>
#include <ranges>

using namespace madevent_py;

namespace {

template<typename F>
std::vector<torch::Tensor> call_torch_impl(
    const std::vector<torch::Tensor>& args,
    FunctionRuntime& func_runtime,
    F run_func
) {
    //TODO: check batch sizes
    auto n_args = func_runtime.function.inputs().size();
    if (args.size() != n_args) {
        throw std::invalid_argument(std::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    std::vector<Tensor> inputs;
    DevicePtr expected_device = nullptr;
    for (int i = 0; i < n_args; ++i) {
        auto& arg = args.at(i);
        auto& input_type = func_runtime.function.inputs().at(i).type;
        auto tensor = torch_to_tensor(arg, input_type, i, expected_device);
        if (i == 0) expected_device = tensor.device();
        inputs.push_back(tensor);
    }

    Runtime* runtime;
    if (expected_device == cpu_device()) {
        if (!func_runtime.cpu_runtime) {
            if (func_runtime.context) {
                if (func_runtime.context->device() != cpu_device()) {
                    throw std::invalid_argument("Given context does not have device CPU");
                }
                func_runtime.cpu_runtime = build_runtime(
                    func_runtime.function, func_runtime.context
                );
            } else {
                func_runtime.cpu_runtime = build_runtime(
                    func_runtime.function, madevent::default_context()
                );
            }
        }
        runtime = func_runtime.cpu_runtime.get();
    } else {
        if (!func_runtime.cuda_runtime) {
            if (func_runtime.context) {
                if (func_runtime.context->device() != cuda_device()) {
                    throw std::invalid_argument("Given context does not have device CUDA");
                }
                func_runtime.cuda_runtime = build_runtime(
                    func_runtime.function, func_runtime.context
                );
            } else {
                //TODO: default cuda context
                func_runtime.cuda_runtime = build_runtime(
                    func_runtime.function, madevent::default_cuda_context()
                );
            }
        }
        runtime = func_runtime.cuda_runtime.get();
    }
    std::vector<torch::Tensor> return_vec;
    for (auto& tensor : run_func(inputs, runtime)) {
        return_vec.push_back(tensor_to_torch(tensor));
    }
    return return_vec;
}

}

torch::Tensor madevent_py::tensor_to_torch(Tensor tensor) {
    std::vector<int64_t> shape {tensor.shape().begin(), tensor.shape().end()};
    std::vector<int64_t> stride {tensor.stride().begin(), tensor.stride().end()};
    if (tensor.dtype() == DataType::batch_sizes) {
        auto& batch_sizes = tensor.batch_sizes();
        torch::Tensor tensor = torch::zeros(
            {static_cast<int64_t>(batch_sizes.size())},
            torch::TensorOptions().dtype(torch::kInt64)
        );
        auto accessor = tensor.accessor<int64_t, 1>();
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
    }
    if (!dtype_ok) {
        throw std::invalid_argument(
            std::format("Argument {}: dtype not accepted", arg_index + 1)
        );
    }
    auto tensor = torch_tensor
        .permute(permutation)
        .to(target_dtype)
        .contiguous()
        .permute(permutation);

    DevicePtr device;
    if (tensor.is_cuda()) {
        device = cuda_device();
    } else {
        device = cpu_device();
    }
    if (expected_device && device != expected_device) {
        throw std::invalid_argument("All inputs have to be on the same device.");
    }
    if (is_batch_sizes) {
        if (n_dims != 1) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected 1, got {}",
                arg_index + 1, n_dims
            ));
        }
        if (tensor.size(0) != expected_type.batch_size_list.size()) {
            throw std::invalid_argument(std::format(
                "Argument {}, dimension 0: shape mismatch. Expected {}, got {}",
                arg_index + 1, expected_type.batch_size_list.size(), tensor.size(0)
            ));
        }
        std::vector<std::size_t> batch_sizes(
            tensor.data_ptr<int64_t>(), tensor.data_ptr<int64_t>() + tensor.numel()
        );
        return {batch_sizes};
    } else if (
        expected_type.batch_size == BatchSize::one &&
        expected_type.shape.size() == 0
    ) {
        switch(expected_type.dtype) {
        case DataType::dt_float:
            return {tensor.item<double>(), device};
        case DataType::dt_int:
            return {tensor.item<int64_t>(), device};
        default:
            throw std::logic_error("unreachable");
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
        }
        return {
            expected_type.dtype, shape, device, data_ptr,
            [tensor] () mutable { tensor = torch::Tensor(); }
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
    }
    SizeVec shape;
    for (std::size_t i = 0; i < n_dims; ++i) {
        shape.push_back(tensor.size(i));
    }
    return {
        dtype, shape, cpu_device(), data_ptr,
        [tensor] () mutable { tensor = torch::Tensor(); }
    };
}

std::vector<torch::Tensor> madevent_py::call_torch(
    FunctionRuntime& func_runtime,
    const std::vector<torch::Tensor>& args
) {
    return call_torch_impl(
        args,
        func_runtime,
        [&] (auto inputs, Runtime* runtime) {
            return runtime->run(inputs);
        }
    );
}

std::tuple<
    std::vector<torch::Tensor>,
    std::vector<std::optional<torch::Tensor>>,
    std::vector<bool>
> madevent_py::call_with_grad_torch(
    FunctionRuntime& func_runtime,
    const std::vector<torch::Tensor>& args,
    const std::vector<bool>& input_requires_grad
) {
    std::vector<std::optional<torch::Tensor>> local_grads;
    std::vector<bool> eval_grad;
    auto outputs = call_torch_impl(
        args,
        func_runtime,
        [&] (auto inputs, Runtime* runtime) {
            auto [out, loc_grad, ev_grad] = runtime->run_with_grad(
                inputs, input_requires_grad
            );
            for (auto& grad : loc_grad) {
                local_grads.push_back(tensor_to_torch_opt(grad));
            }
            eval_grad = ev_grad;
            return out;
        }
    );
    return {outputs, local_grads, eval_grad};
}
std::tuple<
    std::vector<std::optional<torch::Tensor>>,
    std::vector<std::tuple<std::string, std::optional<torch::Tensor>>>
> madevent_py::call_backward_torch(
    FunctionRuntime& func_runtime,
    const std::vector<torch::Tensor>& output_grads,
    const std::vector<std::optional<torch::Tensor>>& stored_locals,
    const std::vector<bool>& eval_grad
) {
    std::vector<Tensor> arg_out;
    for (auto& grad : output_grads) {
        arg_out.push_back(torch_to_tensor_unchecked(grad));
    }
    std::vector<Tensor> arg_locals;
    for (auto& local : stored_locals) {
        arg_locals.push_back(torch_to_tensor_unchecked(local));
    }
    // TODO: checks here
    // TODO: allow for cuda here
    auto [ret_in_grads, ret_glob_grads] = func_runtime.cpu_runtime->run_backward(
        arg_out, arg_locals, eval_grad
    );
    std::vector<std::optional<torch::Tensor>> input_grads;
    for (auto& grad : ret_in_grads) {
        input_grads.push_back(tensor_to_torch_opt(grad));
    }
    std::vector<std::tuple<std::string, std::optional<torch::Tensor>>> global_grads;
    for (auto& [name, grad] : ret_glob_grads) {
        global_grads.push_back({name, tensor_to_torch_opt(grad)});
    }
    return {input_grads, global_grads};
}
