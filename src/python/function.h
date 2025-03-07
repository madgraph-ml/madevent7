#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef TORCH_FOUND
#include <torch/extension.h>
#endif

#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode.h"
#ifdef CUDA_FOUND
#include "madevent/backend/cuda/runtime.h"
#include "madevent/backend/cuda/device.h"
#endif


namespace py = pybind11;
using namespace madevent;

namespace madevent_py {

class FunctionRuntime {
public:
    FunctionRuntime(Function function) : function(function), context(nullptr) {}
    FunctionRuntime(Function function, ContextPtr context) :
        function(function), context(context) {}
    std::vector<py::array_t<double>> call_numpy(std::vector<py::array> args);
#ifdef TORCH_FOUND
    std::vector<torch::Tensor> call_torch(std::vector<torch::Tensor> args);
#endif

private:
    Function function;
    ContextPtr context;
    std::optional<cpu::Runtime> cpu_runtime;
#ifdef CUDA_FOUND
    std::optional<cuda::Runtime> cuda_runtime;
#endif
};

}
