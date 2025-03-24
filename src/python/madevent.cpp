#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "madevent/util.h"
#include "madevent/madcode.h"
#include "madevent/phasespace.h"
#include "madevent/backend/cpu/thread_pool.h"
#include "madevent/driver.h"
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
    InstrCopy(InstructionPtr instr) : name(instr->name()), opcode(instr->opcode()) {}
};

class PyMapping : public Mapping {
public:
    using Mapping::Mapping;

    Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_forward_impl, &fb, inputs, conditions
        );
    }

    Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_inverse_impl, &fb, inputs, conditions
        );
    }
};

class PyFunctionGenerator : public FunctionGenerator {
public:
    using FunctionGenerator::FunctionGenerator;

    ValueVec build_function_impl(
        FunctionBuilder& fb, const ValueVec& args
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            ValueVec, FunctionGenerator, build_function_impl, &fb, &args
        );
    }
};

}

PYBIND11_MODULE(_madevent_py, m) {
    py::enum_<DataType>(m, "DataType")
        .value("bool", DataType::dt_bool)
        .value("int", DataType::dt_int)
        .value("float", DataType::dt_float)
        .value("batch_sizes", DataType::batch_sizes)
        .export_values();

    py::class_<BatchSize>(m, "BatchSize")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        .def_readonly_static("one", &BatchSize::one)
        .def("__str__", &to_string<BatchSize>)
        .def("__repr__", &to_string<BatchSize>);
    m.attr("batch_size") = py::cast(batch_size);

    py::class_<Type>(m, "Type")
        .def(py::init<DataType, BatchSize, std::vector<int>>(),
             py::arg("dtype"), py::arg("batch_size"), py::arg("shape"))
        .def(py::init<std::vector<BatchSize>>(), py::arg("batch_size_list"))
        .def_readonly("dtype", &Type::dtype)
        .def_readonly("batch_size", &Type::batch_size)
        .def_readonly("shape", &Type::shape)
        .def("__str__", &to_string<Type>)
        .def("__repr__", &to_string<Type>);
    m.attr("single_float") = py::cast(single_float);
    m.attr("single_int") = py::cast(single_int);
    m.attr("single_bool") = py::cast(single_bool);
    m.attr("batch_float") = py::cast(batch_float);
    m.attr("batch_int") = py::cast(batch_int);
    m.attr("batch_bool") = py::cast(batch_bool);
    m.attr("batch_four_vec") = py::cast(batch_four_vec);

    py::class_<InstrCopy>(m, "Instruction")
        .def("__str__", [](const InstrCopy& instr) { return instr.name; } )
        .def_readonly("name", &InstrCopy::name)
        .def_readonly("opcode", &InstrCopy::opcode);

    py::class_<Value>(m, "Value")
        .def(py::init<bool>(), py::arg("value"))
        .def(py::init<int64_t>(), py::arg("value"))
        .def(py::init<double>(), py::arg("value"))
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type)
        .def_readonly("literal_value", &Value::literal_value)
        .def_readonly("local_index", &Value::local_index);
    py::implicitly_convertible<bool, Value>();
    py::implicitly_convertible<int64_t, Value>();
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
        .def("store", &Function::store, py::arg("file"))
        .def_static("load", &Function::load, py::arg("file"))
        .def_property_readonly("inputs", &Function::inputs)
        .def_property_readonly("outputs", &Function::outputs)
        .def_property_readonly("locals", &Function::locals)
        .def_property_readonly("globals", &Function::globals)
        .def_property_readonly("instructions", &Function::instructions);

    py::class_<Device, DevicePtr> device(m, "Device");
    m.def("cpu_device", &cpu_device);

    py::class_<MatrixElement>(m, "MatrixElement")
        .def(py::init<const std::string&, const std::string&, std::size_t, double>(),
             py::arg("file"), py::arg("param_card"),
             py::arg("process_index"), py::arg("alpha_s"))
        .def("on_gpu", &MatrixElement::on_gpu)
        .def("particle_count", &MatrixElement::particle_count)
        .def("diagram_count", &MatrixElement::diagram_count);

    py::class_<PdfSet>(m, "PdfSet")
        .def("alpha_s", &PdfSet::alpha_s, py::arg("q2"));

    py::class_<Context, ContextPtr>(m, "Context")
        .def(py::init<>())
        .def(py::init<DevicePtr>(), py::arg("device"))
        .def("load_matrix_element", &Context::load_matrix_element,
             py::arg("file"), py::arg("param_card"),
             py::arg("process_index"), py::arg("alpha_s"))
        .def("load_pdf", &Context::load_pdf, py::arg("name"), py::arg("index")=0)
        .def("define_global", &Context::define_global,
             py::arg("name"), py::arg("dtype"), py::arg("shape"),
             py::arg("requires_grad")=false)
        .def("get_global", &Context::global, py::arg("name"))
        .def("global_requires_grad", &Context::global_requires_grad, py::arg("name"))
        .def("global_exists", &Context::global_exists, py::arg("name"))
        .def("matrix_element", &Context::matrix_element,
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("pdf_set", &Context::pdf_set,
             py::return_value_policy::reference_internal)
        .def("save", &Context::save, py::arg("file"))
        .def("load", &Context::load, py::arg("file"))
        .def("device", &Context::device)
        .def_static("default_context", &Context::default_context);

    py::class_<FunctionRuntime>(m, "FunctionRuntime")
        .def(py::init<Function>(), py::arg("function"))
        .def(py::init<Function, ContextPtr>(), py::arg("function"), py::arg("context"))
#ifdef TORCH_FOUND
        .def("call", &FunctionRuntime::call_torch)
#endif
        .def("call", &FunctionRuntime::call_numpy);

    py::class_<Tensor>(m, "Tensor")
#ifdef TORCH_FOUND
        .def("torch", &tensor_to_torch)
#endif
        .def("numpy", &tensor_to_numpy);

    auto& fb = py::class_<FunctionBuilder>(m, "FunctionBuilder")
        .def(py::init<const std::vector<Type>, const std::vector<Type>>(),
             py::arg("input_types"), py::arg("output_types"))
        .def("input", &FunctionBuilder::input, py::arg("index"))
        .def("input_range", &FunctionBuilder::input_range,
             py::arg("start_index"), py::arg("end_index"))
        .def("output", &FunctionBuilder::output, py::arg("index"), py::arg("value"))
        .def("output_range", &FunctionBuilder::output_range,
             py::arg("start_index"), py::arg("values"))
        .def("get_global", &FunctionBuilder::global,
             py::arg("name"), py::arg("dtype"), py::arg("shape"))
        //.def("instruction", &FunctionBuilder::instruction, py::arg("name"), py::arg("args"))
        .def("function", &FunctionBuilder::function);
    add_instructions(fb);

    py::class_<Mapping, PyMapping>(m, "Mapping", py::dynamic_attr())
        .def(py::init<TypeVec, TypeVec, TypeVec>(),
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
             py::arg("observable"), py::arg("limit_type"),
             py::arg("value"), py::arg("pids"))
        .def_readonly("observable", &Cuts::CutItem::observable)
        .def_readonly("limit_type", &Cuts::CutItem::limit_type)
        .def_readonly("value", &Cuts::CutItem::value)
        .def_readonly("pids", &Cuts::CutItem::pids);
    auto cuts = py::class_<Cuts>(m, "Cuts")
        .def(py::init<std::vector<int>, std::vector<Cuts::CutItem>>(),
             py::arg("pids"), py::arg("cut_data"))
        .def("build_function", &Cuts::build_function,
             py::arg("builder"), py::arg("sqrt_s"), py::arg("momenta"))
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
    py::class_<VegasMapping, Mapping>(m, "VegasMapping")
        .def(py::init<std::size_t, std::size_t, const std::string&>(),
             py::arg("dimension"), py::arg("bin_count"), py::arg("prefix")="")
        .def("grid_name", &VegasMapping::grid_name)
        .def("initialize_global", &VegasMapping::initialize_global, py::arg("context"));

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
        .def("compare", &Topology::compare,
             py::arg("other"), py::arg("compare_t_propagators"))
        .def_readonly("incoming_masses", &Topology::incoming_masses)
        .def_readonly("outgoing_masses", &Topology::outgoing_masses)
        .def_readonly("t_propagators", &Topology::t_propagators)
        .def_readonly("decays", &Topology::decays)
        .def_readonly("permutation", &Topology::permutation)
        .def_readonly("inverse_permutation", &Topology::inverse_permutation)
        .def_readonly("decay_hash", &Topology::decay_hash);
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
        .value("chili", PhaseSpaceMapping::chili)
        .export_values();
    psmap
        .def(py::init<Topology&, double, bool, double, double,
                      PhaseSpaceMapping::TChannelMode, std::optional<Cuts>>(),
             py::arg("topology"), py::arg("s_lab"),
             py::arg("leptonic")=false, py::arg("s_min_epsilon")=1e-2, py::arg("nu")=1.4,
             py::arg("t_channel_mode")=PhaseSpaceMapping::propagator,
             py::arg("cuts")=std::nullopt)
        .def(py::init<const std::vector<double>&, double, bool, double, double,
                      PhaseSpaceMapping::TChannelMode, std::optional<Cuts>>(),
             py::arg("topology"), py::arg("s_lab"),
             py::arg("leptonic")=false, py::arg("s_min_epsilon")=1e-2, py::arg("nu")=1.4,
             py::arg("mode")=PhaseSpaceMapping::rambo,
             py::arg("cuts")=std::nullopt)
        .def("random_dim", &PhaseSpaceMapping::random_dim)
        .def("particle_count", &PhaseSpaceMapping::particle_count);

    py::class_<FastRamboMapping, Mapping>(m, "FastRamboMapping")
        .def(py::init<std::size_t, bool>(), py::arg("n_particles"), py::arg("massless"));
    /*py::class_<MergeOptimizer>(m, "MergeOptimizer")
        .def(py::init<Function&>(), py::arg("function"))
        .def("optimize", &MergeOptimizer::optimize);*/
    py::class_<MultiChannelMapping, Mapping>(m, "MultiChannelMapping")
        .def(py::init<std::vector<PhaseSpaceMapping>&>(), py::arg("mappings"));

    py::class_<FunctionGenerator, PyFunctionGenerator>(
             m, "FunctionGenerator", py::dynamic_attr())
        .def(py::init<TypeVec, TypeVec>(),
             py::arg("arg_types"), py::arg("return_types"))
        .def("function", &FunctionGenerator::function)
        .def("build_function", &FunctionGenerator::build_function,
             py::arg("builder"), py::arg("args"));
    py::class_<DifferentialCrossSection, FunctionGenerator>(m, "DifferentialCrossSection")
        .def(py::init<const std::vector<DifferentialCrossSection::PidOptions>&,
                      double, double, std::size_t, std::vector<int64_t>>(),
             py::arg("pid_options"), py::arg("e_cm2"), py::arg("q2"),
             py::arg("channel_count")=1, py::arg("amp2_remap")=std::vector<int64_t>{})
        .def("pid_options", &DifferentialCrossSection::pid_options);
    py::class_<Unweighter, FunctionGenerator>(m, "Unweighter")
        .def(py::init<const TypeVec&, std::size_t>(),
             py::arg("types"), py::arg("particle_count"));
    py::class_<Integrand, FunctionGenerator>(m, "Integrand")
        .def(py::init<const PhaseSpaceMapping&, const DifferentialCrossSection&, int>(),
             py::arg("mapping"), py::arg("diff_xs"), py::arg("flags")=0)
        .def("particle_count", &Integrand::particle_count)
        .def("flags", &Integrand::flags)
        .def_readonly_static("sample", &Integrand::sample)
        .def_readonly_static("unweight", &Integrand::unweight)
        .def_readonly_static("return_momenta", &Integrand::return_momenta)
        .def_readonly_static("return_x1_x2", &Integrand::return_x1_x2)
        .def_readonly_static("return_random", &Integrand::return_random);

    py::class_<EventGenerator::Config>(m, "EventGeneratorConfig")
        .def(py::init<>())
        .def_readwrite("target_count", &EventGenerator::Config::target_count)
        .def_readwrite("vegas_damping", &EventGenerator::Config::vegas_damping)
        .def_readwrite("max_overweight_fraction",
                       &EventGenerator::Config::max_overweight_fraction)
        .def_readwrite("max_overweight_truncation",
                       &EventGenerator::Config::max_overweight_truncation)
        .def_readwrite("start_batch_size", &EventGenerator::Config::start_batch_size)
        .def_readwrite("max_batch_size", &EventGenerator::Config::max_batch_size)
        .def_readwrite("survey_min_iters", &EventGenerator::Config::survey_min_iters)
        .def_readwrite("survey_max_iters", &EventGenerator::Config::survey_max_iters)
        .def_readwrite("survey_target_precision",
                       &EventGenerator::Config::survey_target_precision)
        .def_readwrite("optimization_patience",
                       &EventGenerator::Config::optimization_patience)
        .def_readwrite("optimization_threshold",
                       &EventGenerator::Config::optimization_threshold);
    py::class_<EventGenerator::Status>(m, "EventGeneratorStatus")
        .def(py::init<>())
        .def_readwrite("index", &EventGenerator::Status::index)
        .def_readwrite("mean", &EventGenerator::Status::mean)
        .def_readwrite("error", &EventGenerator::Status::error)
        .def_readwrite("rel_std_dev", &EventGenerator::Status::rel_std_dev)
        .def_readwrite("count", &EventGenerator::Status::count)
        .def_readwrite("count_unweighted", &EventGenerator::Status::count_unweighted)
        .def_readwrite("count_target", &EventGenerator::Status::count_target)
        .def_readwrite("iterations", &EventGenerator::Status::iterations)
        .def_readwrite("done", &EventGenerator::Status::done);
    py::class_<EventGenerator>(m, "EventGenerator")
        .def(py::init<ContextPtr, const std::vector<Integrand>&,
                      const std::string&, const EventGenerator::Config&,
                      const std::optional<std::string>&>(),
             py::arg("context"), py::arg("channels"), py::arg("file_name"),
             py::arg("default_config")=EventGenerator::Config{},
             py::arg("temp_file_dir")=std::nullopt)
        .def("survey", &EventGenerator::survey)
        .def("generate", &EventGenerator::generate)
        .def("status", &EventGenerator::status)
        .def("channel_status", &EventGenerator::channel_status);

    py::class_<VegasGridOptimizer>(m, "VegasGridOptimizer")
#ifdef TORCH_FOUND
        .def("optimize", [](VegasGridOptimizer& opt, torch::Tensor weights, torch::Tensor inputs) {
                opt.optimize(
                    torch_to_tensor(weights, batch_float, 0),
                    torch_to_tensor(inputs, batch_float_array(opt.input_dim()), 1)
                );
             }, py::arg("weights"), py::arg("inputs"))
#endif
        .def(py::init<ContextPtr, const std::string&, double>(),
             py::arg("context"), py::arg("grid_name"), py::arg("damping"));

    m.def("optimize_constants", &optimize_constants, py::arg("function"));
    m.def("set_thread_count", &cpu::ThreadPool::set_thread_count, py::arg("new_count"));
    m.def("format_si_prefix", &format_si_prefix, py::arg("value"));
    m.def("format_with_error", &format_with_error, py::arg("value"), py::arg("error"));
    m.def("format_progress", &format_progress, py::arg("progress"), py::arg("width"));
    m.def("initialize_vegas_grid", &initialize_vegas_grid,
          py::arg("context"), py::arg("grid_name"));
}
