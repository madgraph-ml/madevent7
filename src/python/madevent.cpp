#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "madevent/madcode.h"
#include "madevent/phasespace.h"
#include "instruction_set.h"
#include "function.h"

namespace py = pybind11;
using namespace madevent;
using namespace madevent_py;

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
    InstrCopy(InstructionPtr instr) : name(instr->name), opcode(instr->opcode) {}
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
        .def(py::init<bool>(), py::arg("value"))
        .def(py::init<long long>(), py::arg("value"))
        .def(py::init<double>(), py::arg("value"))
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type)
        .def_readonly("literal_value", &Value::literal_value)
        .def_readonly("local_index", &Value::local_index);
    py::implicitly_convertible<bool, Value>();
    py::implicitly_convertible<long long, Value>();
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

    py::class_<Function>(m, "Function", py::dynamic_attr())
        .def("__str__", &to_string<Function>)
        .def("__repr__", &to_string<Function>)
        .def_readonly("inputs", &Function::inputs)
        .def_readonly("outputs", &Function::outputs)
        .def_readonly("locals", &Function::locals)
        .def_readonly("instructions", &Function::instructions);

    py::class_<FunctionRuntime>(m, "FunctionRuntime")
        .def(py::init<Function>(), py::arg("function"))
#ifdef TORCH_FOUND
        .def("call", &FunctionRuntime::call_torch)
#endif
        .def("call", &FunctionRuntime::call_numpy);

    auto& fb = py::class_<FunctionBuilder>(m, "FunctionBuilder")
        .def(py::init<const std::vector<Type>, const std::vector<Type>>(),
             py::arg("input_types"), py::arg("output_types"))
        .def("input", &FunctionBuilder::input, py::arg("index"))
        .def("input_range", &FunctionBuilder::input_range,
             py::arg("start_index"), py::arg("end_index"))
        .def("output", &FunctionBuilder::output, py::arg("index"), py::arg("value"))
        .def("output_range", &FunctionBuilder::output_range,
             py::arg("start_index"), py::arg("values"))
        //.def("instruction", &FunctionBuilder::instruction, py::arg("name"), py::arg("args"))
        .def("function", &FunctionBuilder::function);
    add_instructions(fb);

    py::class_<Mapping, PyMapping>(m, "Mapping", py::dynamic_attr())
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
    py::class_<Propagator>(m, "Propagator")
        .def(py::init<double, double>(), py::arg("mass"), py::arg("width"))
        .def_readonly("mass", &Propagator::mass)
        .def_readonly("width", &Propagator::width);
    py::class_<TPropagatorMapping, Mapping>(m, "TPropagatorMapping")
        .def(py::init<std::vector<Propagator>, double, bool>(),
             py::arg("propagators"), py::arg("nu")=0., py::arg("map_resonances")=false);
    py::class_<Cuts::CutItem>(m, "CutItem")
        .def(py::init<Cuts::CutObservable, Cuts::LimitType, double, Cuts::PidVec>(),
             py::arg("observable"), py::arg("limit_type"), py::arg("value"), py::arg("pids"))
        .def_readonly("observable", &Cuts::CutItem::observable)
        .def_readonly("limit_type", &Cuts::CutItem::limit_type)
        .def_readonly("value", &Cuts::CutItem::value)
        .def_readonly("pids", &Cuts::CutItem::pids);
    auto cuts = py::class_<Cuts>(m, "Cuts")
        .def(py::init<std::vector<int>, std::vector<Cuts::CutItem>>(),
             py::arg("pids"), py::arg("cut_data"))
        .def("build_function", &Cuts::build_function,
             py::arg("builder"), py::arg("sqrt_s"), py::arg("momenta"), py::arg("weight"))
        .def("get_sqrt_s_min", &Cuts::get_sqrt_s_min)
        .def("get_eta_max", &Cuts::get_eta_max)
        .def("get_pt_min", &Cuts::get_pt_min)
        .def_readonly_static("jet_pids", &Cuts::jet_pids)
        .def_readonly_static("bottom_pids", &Cuts::bottom_pids)
        .def_readonly_static("lepton_pids", &Cuts::lepton_pids)
        .def_readonly_static("missing_pids", &Cuts::missing_pids)
        .def_readonly_static("photon_pids", &Cuts::photon_pids);
    py::enum_<Cuts::CutObservable>(cuts, "CutObservable")
        .value("obs_pt", Cuts::obs_pt)
        .value("obs_eta", Cuts::obs_eta)
        .value("obs_dr", Cuts::obs_dr)
        .value("obs_mass", Cuts::obs_mass)
        .value("obs_sqrt_s", Cuts::obs_sqrt_s)
        .export_values();
    py::enum_<Cuts::LimitType>(cuts, "LimitType")
        .value("min", Cuts::min)
        .value("max", Cuts::max)
        .export_values();

    py::class_<Diagram::LineRef>(m, "LineRef")
        .def(py::init<std::string>(), py::arg("str"))
        .def("__repr__", &to_string<Diagram::LineRef>);
    py::implicitly_convertible<std::string, Diagram::LineRef>();
    py::class_<Diagram>(m, "Diagram")
        .def(py::init<std::vector<double>&,
                      std::vector<double>&,
                      std::vector<Propagator>&,
                      std::vector<Diagram::Vertex>&>(),
             py::arg("incoming_masses"),
             py::arg("outgoing_masses"),
             py::arg("propagators"),
             py::arg("vertices"))
        .def_readonly("incoming_vertices", &Diagram::incoming_vertices)
        .def_readonly("outgoing_vertices", &Diagram::outgoing_vertices)
        .def_readonly("propagator_vertices", &Diagram::propagator_vertices)
        .def_readonly("t_propagators", &Diagram::t_propagators)
        .def_readonly("t_vertices", &Diagram::t_vertices)
        .def_readonly("lines_after_t", &Diagram::lines_after_t)
        .def_readonly("decays", &Diagram::decays);
    auto& topology = py::class_<Topology>(m, "Topology")
        .def(py::init<Diagram&, Topology::DecayMode>(),
             py::arg("diagram"), py::arg("decay_mode"))
        .def("compare", &Topology::compare, py::arg("other"), py::arg("compare_t_propagators"))
        .def_readonly("incoming_masses", &Topology::incoming_masses)
        .def_readonly("outgoing_masses", &Topology::outgoing_masses)
        .def_readonly("t_propagators", &Topology::t_propagators)
        .def_readonly("decays", &Topology::decays)
        .def_readonly("permutation", &Topology::permutation)
        .def_readonly("inverse_permutation", &Topology::inverse_permutation);
    py::enum_<Topology::DecayMode>(topology, "DecayMode")
        .value("no_decays", Topology::no_decays)
        .value("massive_decays", Topology::massive_decays)
        .value("all_decays", Topology::all_decays)
        .export_values();
    py::enum_<Topology::ComparisonResult>(topology, "ComparisonResult")
        .value("equal", Topology::equal)
        .value("permuted", Topology::permuted)
        .value("different", Topology::different)
        .export_values();
    py::class_<Topology::Decay>(m, "Decay")
        .def_readonly("propagator", &Topology::Decay::propagator)
        .def_readonly("child_count", &Topology::Decay::child_count);
    py::class_<PhaseSpaceMapping, Mapping> psmap(m, "PhaseSpaceMapping");
    py::enum_<PhaseSpaceMapping::TChannelMode>(psmap, "TChannelMode")
        .value("propagator", PhaseSpaceMapping::propagator)
        .value("rambo", PhaseSpaceMapping::rambo)
        .export_values();
    psmap.def(py::init<Topology&, double, /*double,*/ bool, double, double,
                       PhaseSpaceMapping::TChannelMode, std::optional<Cuts>>(),
              py::arg("topology"), py::arg("s_lab"), //py::arg("s_hat_min")=0.0,
              py::arg("leptonic")=false, py::arg("s_min_epsilon")=1e-2, py::arg("nu")=1.4,
              py::arg("t_channel_mode")=PhaseSpaceMapping::propagator,
              py::arg("cuts")=std::nullopt);
    py::class_<FastRamboMapping, Mapping>(m, "FastRamboMapping")
        .def(py::init<std::size_t, bool>(), py::arg("n_particles"), py::arg("massless"));
    py::class_<MergeOptimizer>(m, "MergeOptimizer")
        .def(py::init<Function&>(), py::arg("function"))
        .def("optimize", &MergeOptimizer::optimize);
    py::class_<MultiChannelMapping, Mapping>(m, "MultiChannelMapping")
        .def(py::init<std::vector<PhaseSpaceMapping>&>(), py::arg("mappings"));

    m.def("optimize_constants", &optimize_constants);
}
