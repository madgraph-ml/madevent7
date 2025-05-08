#include "function_runtime.h"

#include <stdexcept>
#include <format>
#include <ranges>

#include "dlpack.h"

using namespace madevent_py;
using namespace pybind11::literals;

namespace {

struct ManagerContext {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    std::vector<int64_t> batch_sizes;
    Tensor tensor;
};

void deleter(struct DLManagedTensor* self) {
    delete static_cast<ManagerContext*>(self->manager_ctx);
    delete self;
};

std::tuple<std::vector<Tensor>, Runtime*> check_and_convert_args(
    const std::vector<py::object>& args,
    FunctionRuntime& func_runtime
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
        auto tensor = dlpack_to_tensor(arg, input_type, i, expected_device);
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
    return {inputs, runtime};
}

}

std::tuple<int, int> madevent_py::dlpack_device(Tensor tensor) {
    return {tensor.device() == cpu_device() ? kDLCPU : kDLCUDA, 0};
}

py::object madevent_py::tensor_to_dlpack(Tensor tensor) {
    if (!tensor) return py::none();

    DLManagedTensor* dl_tensor;
    if (tensor.dtype() == DataType::batch_sizes) {
        ManagerContext* context = new ManagerContext{
            {tensor.shape().begin(), tensor.shape().end()},
            {tensor.stride().begin(), tensor.stride().end()},
            {tensor.batch_sizes().begin(), tensor.batch_sizes().end()},
            {}
        };
        dl_tensor = new DLManagedTensor{
            {
                static_cast<void*>(context->batch_sizes.data()),
                {kDLCPU, 0},
                static_cast<int32_t>(context->shape.size()),
                {kDLFloat, 64, 1},
                context->shape.data(),
                context->stride.data(),
                0
            },
            static_cast<void*>(context),
            &deleter
        };
    } else {
        DLDataType dtype;
        switch(tensor.dtype()) {
            case DataType::dt_float: dtype = {kDLFloat, 64, 1}; break;
            case DataType::dt_int: dtype = {kDLInt, 64, 1}; break;
            default: break;
        }
        ManagerContext* context = new ManagerContext{
            {tensor.shape().begin(), tensor.shape().end()},
            {tensor.stride().begin(), tensor.stride().end()},
            {},
            tensor
        };
        dl_tensor = new DLManagedTensor{
            {
                context->tensor.data(),
                {tensor.device() == cpu_device() ? kDLCPU : kDLCUDA, 0},
                static_cast<int32_t>(context->shape.size()),
                dtype,
                context->shape.data(),
                context->stride.data(),
                tensor.offset() * tensor.dtype_size()
            },
            static_cast<void*>(context),
            &deleter
        };
    }

    return py::capsule(
        dl_tensor,
        "dltensor",
        [](PyObject* self) {
            // Implement capsule deleter following the example in
            // https://dmlc.github.io/dlpack/latest/python_spec.html
            if (PyCapsule_IsValid(self, "used_dltensor")) {
                return;
            }
            DLManagedTensor *managed = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(self, "dltensor")
            );
            if (managed == NULL) {
                PyErr_WriteUnraisable(self);
                return;
            }
            if (managed->deleter) {
                managed->deleter(managed);
            }
        }
    );
}

Tensor madevent_py::dlpack_to_tensor(
    py::object tensor,
    std::optional<Type> expected_type,
    std::size_t arg_index,
    DevicePtr expected_device
) {
    if (tensor.is_none()) return {};

    py::object dlpack_func = tensor.attr("__dlpack__");
    py::object capsule_obj;
    try {
        capsule_obj = dlpack_func(
            "max_version"_a=std::make_tuple(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)
        );
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_TypeError)) {
            capsule_obj = dlpack_func();
        } else {
            throw;
        }
    }
    PyObject* capsule = capsule_obj.ptr();
    if (!capsule) throw std::runtime_error("value must support the dlpack protocol");

    void* managed_ptr;
    DLTensor* dl_tensor;
    bool versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    if (versioned) {
        managed_ptr = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        if (!managed_ptr) throw std::runtime_error("value must support the dlpack protocol");
        auto managed = static_cast<DLManagedTensorVersioned*>(managed_ptr);
        if (managed->version.major > 1) {
            throw std::runtime_error("unsupported dlpack version");
        }
        dl_tensor = &managed->dl_tensor;
    } else {
        managed_ptr = PyCapsule_GetPointer(capsule, "dltensor");
        if (!managed_ptr) throw std::runtime_error("value must support the dlpack protocol");
        auto managed = static_cast<DLManagedTensor*>(managed_ptr);
        dl_tensor = &managed->dl_tensor;
    }

    bool is_batch_sizes = expected_type ?
        false : expected_type->dtype == DataType::batch_sizes;
    DataType dtype;
    if (
        dl_tensor->dtype.code == kDLFloat &&
        dl_tensor->dtype.bits == 64 &&
        dl_tensor->dtype.lanes == 1
    ) {
        dtype = DataType::dt_float;
        if (expected_type && expected_type->dtype != DataType::dt_float) {
            throw std::invalid_argument(
                std::format("Argument {}: expected dtype float", arg_index + 1)
            );
        }
    } else if (
        dl_tensor->dtype.code == kDLInt &&
        dl_tensor->dtype.bits == 64 &&
        dl_tensor->dtype.lanes == 1
    ) {
        dtype = DataType::dt_int;
        if (expected_type && expected_type->dtype != DataType::dt_int && !is_batch_sizes) {
            throw std::invalid_argument(
                std::format("Argument {}: expected dtype int", arg_index + 1)
            );
        }
    } else {
        throw std::invalid_argument(std::format(
            "Argument {}: input dtype must be 64-bit float or int", arg_index + 1
        ));
    }

    DevicePtr device;
    if (dl_tensor->device.device_type == kDLCUDA && dl_tensor->device.device_id == 0) {
        device = cuda_device();
    } else if (dl_tensor->device.device_type == kDLCPU && dl_tensor->device.device_id == 0) {
        device = cpu_device();
    } else {
        throw std::invalid_argument(
            std::format("Argument {}: device not supported", arg_index + 1)
        );
    }
    if (expected_device && device != expected_device) {
        throw std::invalid_argument("All inputs have to be on the same device.");
    }

    Tensor ret_tensor;
    if (is_batch_sizes) {
        if (dl_tensor->ndim != 1) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected 1, got {}",
                arg_index + 1, dl_tensor->ndim
            ));
        }
        std::size_t count = dl_tensor->shape[0];
        if (count != expected_type->batch_size_list.size()) {
            throw std::invalid_argument(std::format(
                "Argument {}, dimension 0: shape mismatch. Expected {}, got {}",
                arg_index + 1, expected_type->batch_size_list.size(), dl_tensor->shape[0]
            ));
        }
        if (dl_tensor->device.device_type != kDLCPU || dl_tensor->device.device_id != 0) {
            throw std::invalid_argument(std::format(
                "Argument {}: batch size list must have device CPU", arg_index + 1
            ));
        }
        std::vector<std::size_t> batch_sizes(count);
        int64_t* data_ptr = reinterpret_cast<int64_t*>(
            static_cast<uint8_t*>(dl_tensor->data) + dl_tensor->byte_offset
        );
        std::size_t bs_stride = dl_tensor->strides ? dl_tensor->strides[0] : 1;
        for (std::size_t i = 0; i < count; ++i) {
            batch_sizes[i] = data_ptr[bs_stride * i];
        }
        if (versioned) {
            auto ptr = static_cast<DLManagedTensorVersioned*>(managed_ptr);
            if (ptr->deleter) ptr->deleter(ptr);
        } else {
            auto ptr = static_cast<DLManagedTensor*>(managed_ptr);
            if (ptr->deleter) ptr->deleter(ptr);
        }
        ret_tensor = {batch_sizes};
    /*} else if (
        expected_type &&
        expected_type->batch_size == BatchSize::one &&
        expected_type->shape.size() == 0
    ) {
        switch(expected_type->dtype) {
        case DataType::dt_float:
            return {tensor.item<double>(), device};
        case DataType::dt_int:
            return {tensor.item<int64_t>(), device};
        default:
            throw std::logic_error("unreachable");
        }*/
    } else {
        bool is_batch = !expected_type || expected_type->batch_size != BatchSize::one;
        if (expected_type && dl_tensor->ndim != expected_type->shape.size() + is_batch) {
            throw std::invalid_argument(std::format(
                "Argument {}: wrong input dimension. Expected {}, got {}",
                arg_index + 1, expected_type->shape.size() + 1, dl_tensor->ndim
            ));
        }
        std::vector<size_t> shape, stride;
        if (!is_batch) {
            shape.push_back(1);
            stride.push_back(1);
        }
        shape.insert(shape.end(), dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim);
        if (dl_tensor->strides) {
            stride.insert(
                stride.end(), dl_tensor->strides, dl_tensor->strides + dl_tensor->ndim
            );
        } else {
            stride.resize(shape.size());
            std::size_t stride_prod = 1;
            for (auto [size_i, stride_i] : zip(
                std::views::reverse(shape), std::views::reverse(stride)
            )) {
                stride_i = stride_prod;
                stride_prod *= size_i;
            }
        }
        if (expected_type) {
            for (int j = is_batch; j < dl_tensor->ndim; ++j) {
                if (shape.at(j) != expected_type->shape.at(j - is_batch)) {
                    throw std::invalid_argument(std::format(
                        "Argument {}, dimension {}: shape mismatch. Expected {}, got {}",
                        arg_index + 1, j, expected_type->shape.at(j - is_batch), shape.at(j)
                    ));
                }
            }
        }
        void* data_ptr = static_cast<void*>(
            static_cast<uint8_t*>(dl_tensor->data) + dl_tensor->byte_offset
        );
        std::function<void()> deleter;
        if (versioned) {
            deleter = [managed_ptr] () {
                auto ptr = static_cast<DLManagedTensorVersioned*>(managed_ptr);
                if (ptr->deleter) ptr->deleter(ptr);
            };
        } else {
            deleter = [managed_ptr] () {
                auto ptr = static_cast<DLManagedTensor*>(managed_ptr);
                if (ptr->deleter) ptr->deleter(ptr);
            };
        }
        ret_tensor = {dtype, shape, stride, device, data_ptr, deleter};
    }

    if (PyCapsule_SetName(
        capsule, versioned ? "used_dltensor_versioned" : "used_dltensor"
    ) < 0) {
        throw std::runtime_error("could not rename capsule");
    }
    return ret_tensor;
}

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

std::vector<Tensor> FunctionRuntime::call(std::vector<py::object> args) {
    auto [inputs, runtime] = check_and_convert_args(args, *this);
    return runtime->run(inputs);
}

std::tuple<
    std::vector<Tensor>,
    std::vector<std::optional<Tensor>>,
    std::vector<bool>
> FunctionRuntime::call_with_grad(
    const std::vector<py::object>& args,
    const std::vector<bool>& input_requires_grad
) {
    auto [inputs, runtime] = check_and_convert_args(args, *this);
    auto [outputs, loc_grad, eval_grad] = runtime->run_with_grad(
        inputs, input_requires_grad
    );
    std::vector<std::optional<Tensor>> local_grads;
    for (auto& grad : loc_grad) {
        if (grad) {
            local_grads.push_back(grad);
        } else {
            local_grads.push_back({});
        }
    }
    return {outputs, local_grads, eval_grad};
}

std::tuple<
    std::vector<std::optional<Tensor>>,
    std::vector<std::tuple<std::string, std::optional<Tensor>>>
> FunctionRuntime::call_backward(
    const std::vector<py::object>& output_grads,
    const std::vector<py::object>& stored_locals,
    const std::vector<bool>& eval_grad
) {
    std::vector<Tensor> arg_out;
    for (auto& grad : output_grads) {
        arg_out.push_back(dlpack_to_tensor(grad));
    }
    std::vector<Tensor> arg_locals;
    for (auto& local : stored_locals) {
        arg_locals.push_back(dlpack_to_tensor(local));
    }
    // TODO: checks here
    // TODO: allow for cuda here
    auto [ret_in_grads, ret_glob_grads] = cpu_runtime->run_backward(
        arg_out, arg_locals, eval_grad
    );
    std::vector<std::optional<Tensor>> input_grads;
    for (auto& grad : ret_in_grads) {
        if (grad) {
            input_grads.push_back(grad);
        } else {
            input_grads.push_back({});
        }
    }
    std::vector<std::tuple<std::string, std::optional<Tensor>>> global_grads;
    for (auto& [name, grad] : ret_glob_grads) {
        if (grad) {
            global_grads.push_back({name, grad});
        } else {
            global_grads.push_back({name, std::nullopt});
        }
    }
    return {input_grads, global_grads};
}
