#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef TORCH_FOUND
#include <torch/extension.h>
#endif

#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode.h"

namespace py = pybind11;
using namespace madevent;

namespace madevent_py {

class FunctionRuntime {
public:
    FunctionRuntime(Function _function) : function(_function) {}
    std::vector<py::array_t<double>> call_numpy(std::vector<py::array> args);
#ifdef TORCH_FOUND
    std::vector<torch::Tensor> call_torch(std::vector<torch::Tensor> args);
#endif

private:
    Function function;
    std::optional<cpu::Runtime> cpu_runtime;
};

}
