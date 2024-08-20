#pragma once

#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fmt/core.h>

#include "madevent/backend/cpu/runtime.h"
#include "madevent/madcode.h"

using namespace madevent;
namespace py = pybind11;

//namespace {

std::vector<py::array_t<double>> run_function(const cpu::Runtime& runtime, const Function& function, py::tuple args) {
    std::vector<cpu::Tensor> inputs;
    auto n_args = function.inputs.size();
    if (args.size() != n_args) {
        throw std::invalid_argument(fmt::format(
            "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
        ));
    }
    using Arr = py::array_t<double, py::array::f_style | py::array::forcecast>;
    std::vector<Arr> arrays;
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
            shape.push_back(arr.shape(j));
        }
        inputs.emplace_back(DT_FLOAT, shape, std::shared_ptr<uint8_t[]>(
            reinterpret_cast<uint8_t*>(arr.mutable_data()), [](auto ptr) {}
        ));
        arrays.push_back(arr);
    }

    auto outputs = runtime.run(inputs);
    std::vector<py::array_t<double>> outputs_numpy;
    for (auto& output : outputs) {
        //auto data_raw = static_cast<void*>(output.data.get());
        auto data_raw = reinterpret_cast<double*>(output.data.get());
        py::capsule destroy(
            new cpu::Tensor::DataPtr(output.data),
            [](void* ptr) { delete static_cast<cpu::Tensor::DataPtr*>(ptr); }
        );
        outputs_numpy.emplace_back(output.shape, output.stride, data_raw, destroy);
    }
    return outputs_numpy;
}

//}
