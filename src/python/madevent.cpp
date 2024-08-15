#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "madevent/madcode.h"
#include "madevent/phasespace.h"
#include "instruction_set.h"

namespace py = pybind11;
using namespace madevent;

namespace {

template<typename T>
auto to_string(const T& object) {
    std::ostringstream str;
    str << object;
    return str.str();
}

struct InstrCopy {
    std::string name;
    int opcode;
    InstrCopy(const InstructionPtr& instr) : name(instr->name), opcode(instr->opcode) {}
};

class PyMapping : public Mapping {
public:
    using Mapping::Mapping;

    Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(Result, Mapping, build_forward_impl, &fb, inputs, conditions);
    }

    Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(Result, Mapping, build_inverse_impl, &fb, inputs, conditions);
    }
};

}

PYBIND11_MODULE(madevent_py, m) {
    py::enum_<DataType>(m, "DataType")
        .value("bool", DataType::DT_BOOL)
        .value("int", DataType::DT_INT)
        .value("float", DataType::DT_FLOAT)
        .export_values();

    py::class_<Type>(m, "Type")
        .def(py::init<DataType, std::vector<int>>(), py::arg("dtype"), py::arg("shape"))
        .def_readonly("dtype", &Type::dtype)
        .def_readonly("shape", &Type::shape);

    m.attr("scalar") = py::cast(scalar);
    m.attr("scalar_int") = py::cast(scalar_int);
    m.attr("scalar_bool") = py::cast(scalar_bool);
    m.attr("four_vector") = py::cast(four_vector);

    py::class_<InstrCopy>(m, "Instruction")
        .def("__str__", [](const InstrCopy& instr) { return instr.name; } )
        .def_readonly("name", &InstrCopy::name)
        .def_readonly("opcode", &InstrCopy::opcode);

    py::class_<Value>(m, "Value")
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type)
        .def_readonly("literal_value", &Value::literal_value)
        .def_readonly("local_index", &Value::local_index);
    py::implicitly_convertible<bool, Value>();
    py::implicitly_convertible<int, Value>();
    py::implicitly_convertible<double, Value>();
    py::implicitly_convertible<std::string, Value>();

    py::class_<InstructionCall>(m, "InstructionCall")
        .def("__str__", &to_string<InstructionCall>)
        .def("__repr__", &to_string<InstructionCall>)
        .def_property_readonly("instruction", [](const InstructionCall& call) -> InstrCopy {
            return call.instruction;
        })
        .def_readonly("inputs", &InstructionCall::inputs)
        .def_readonly("outputs", &InstructionCall::outputs);

    py::class_<Function>(m, "Function")
        .def("__str__", &to_string<Function>)
        .def("__repr__", &to_string<Function>)
        .def_readonly("inputs", &Function::inputs)
        .def_readonly("outputs", &Function::outputs)
        .def_readonly("locals", &Function::locals)
        .def_readonly("instructions", &Function::instructions);

    auto& fb = py::class_<FunctionBuilder>(m, "FunctionBuilder")
        .def(py::init<const std::vector<Type>, const std::vector<Type>>(),
             py::arg("input_types"), py::arg("output_types"))
        .def("input", &FunctionBuilder::input, py::arg("index"))
        .def("input_range", &FunctionBuilder::input_range,
             py::arg("start_index"), py::arg("end_index"))
        .def("output", &FunctionBuilder::output, py::arg("index"), py::arg("value"))
        .def("output_range", &FunctionBuilder::output_range,
             py::arg("start_index"), py::arg("values"))
        .def("instruction", &FunctionBuilder::instruction, py::arg("name"), py::arg("args"))
        .def("function", &FunctionBuilder::function);
    add_instructions(fb);

    py::class_<Mapping, PyMapping>(m, "Mapping")
        .def(py::init<TypeList, TypeList, TypeList>(),
             py::arg("input_types"), py::arg("output_types"), py::arg("condition_types"))
        .def("forward_function", &Mapping::forward_function)
        .def("inverse_function", &Mapping::inverse_function)
        .def("build_forward", &Mapping::build_forward,
             py::arg("builder"), py::arg("inputs"), py::arg("conditions"))
        .def("build_inverse", &Mapping::build_inverse,
             py::arg("builder"), py::arg("inputs"), py::arg("conditions"));

    py::class_<Invariant, Mapping>(m, "Invariant")
        .def(py::init<double, double, double>(),
             py::arg("nu")=0., py::arg("mass")=0., py::arg("width")=0.);
    py::class_<Luminosity, Mapping>(m, "Luminosity")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("s_lab"), py::arg("s_hat_min"), py::arg("s_hat_max")=0.,
             py::arg("nu")=0., py::arg("mass")=0., py::arg("width")=0.);
    py::class_<TwoParticle, Mapping>(m, "TwoParticle")
        .def(py::init<bool>(), py::arg("com"));
    py::class_<TInvariantTwoParticle, Mapping>(m, "TInvariantTwoParticle")
        .def(py::init<bool, double, double, double>(),
             py::arg("com"), py::arg("nu")=0., py::arg("mass")=0., py::arg("width")=0.);
}
