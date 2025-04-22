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

py::array_t<double> tensor_to_numpy(Tensor tensor);

struct FunctionRuntime {
    FunctionRuntime(Function function) : function(function), context(nullptr) {}
    FunctionRuntime(Function function, ContextPtr context) :
        function(function), context(context) {}
    std::vector<py::array_t<double>> call_numpy(std::vector<py::array> args);

    Function function;
    ContextPtr context;
    RuntimePtr cpu_runtime;
    RuntimePtr cuda_runtime;
};

}
