#pragma once

#include <vector>
#include <torch/extension.h>

#include "madevent/madcode.h"
#include "madevent/runtime/runtime_base.h"
#include "function_runtime.h"

using namespace madevent;

namespace madevent_py {

torch::Tensor tensor_to_torch(Tensor tensor);
std::optional<torch::Tensor> tensor_to_torch_opt(Tensor tensor);
Tensor torch_to_tensor(
    torch::Tensor tensor,
    Type expected_type,
    std::size_t arg_index,
    DevicePtr expected_device = nullptr
);
Tensor torch_to_tensor_unchecked(std::optional<torch::Tensor> torch_tensor);

std::vector<torch::Tensor> call_torch(
    FunctionRuntime& func_wrapper, const std::vector<torch::Tensor>& args
);

std::tuple<
    std::vector<torch::Tensor>,
    std::vector<std::optional<torch::Tensor>>,
    std::vector<bool>
> call_with_grad_torch(
    FunctionRuntime& func_wrapper,
    const std::vector<torch::Tensor>& args,
    const std::vector<bool>& input_requires_grad
);

std::tuple<
    std::vector<std::optional<torch::Tensor>>,
    std::vector<std::tuple<std::string, std::optional<torch::Tensor>>>
> call_backward_torch(
    FunctionRuntime& func_wrapper,
    const std::vector<torch::Tensor>& output_grads,
    const std::vector<std::optional<torch::Tensor>>& stored_locals,
    const std::vector<bool>& eval_grad
);

}
