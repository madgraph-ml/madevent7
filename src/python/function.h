#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef TORCH_FOUND
#include <torch/extension.h>
#endif

#include "madevent/madcode.h"
#include "madevent/runtime/runtime_base.h"


namespace py = pybind11;
using namespace madevent;

namespace madevent_py {

#ifdef TORCH_FOUND
torch::Tensor tensor_to_torch(Tensor tensor);
std::optional<torch::Tensor> tensor_to_torch_opt(Tensor tensor);
Tensor torch_to_tensor(
    torch::Tensor tensor,
    Type expected_type,
    std::size_t arg_index,
    DevicePtr expected_device = nullptr
);
Tensor torch_to_tensor_unchecked(std::optional<torch::Tensor> torch_tensor);
#endif
py::array_t<double> tensor_to_numpy(Tensor tensor);

class FunctionRuntime {
public:
    FunctionRuntime(Function function) : function(function), context(nullptr) {}
    FunctionRuntime(Function function, ContextPtr context) :
        function(function), context(context) {}
    std::vector<py::array_t<double>> call_numpy(std::vector<py::array> args);
#ifdef TORCH_FOUND
    std::vector<torch::Tensor> call_torch(const std::vector<torch::Tensor>& args);
    std::tuple<
        std::vector<torch::Tensor>,
        std::vector<std::optional<torch::Tensor>>,
        std::vector<bool>
    > call_with_grad_torch(
        const std::vector<torch::Tensor>& args,
        const std::vector<bool>& input_requires_grad
    );
    std::tuple<
        std::vector<std::optional<torch::Tensor>>,
        std::vector<std::tuple<std::string, std::optional<torch::Tensor>>>
    > call_backward_torch(
        const std::vector<torch::Tensor>& output_grads,
        const std::vector<std::optional<torch::Tensor>>& stored_locals,
        const std::vector<bool>& eval_grad
    );
#endif

private:
    Function function;
    ContextPtr context;
    RuntimePtr cpu_runtime;
    RuntimePtr cuda_runtime;
};

}
