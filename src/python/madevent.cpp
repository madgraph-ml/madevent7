#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "madevent/util.h"
#include "madevent/madcode.h"
#include "madevent/phasespace.h"
#include "madevent/runtime.h"
#include "instruction_set.h"
#include "function_runtime.h"

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

class PyMapping : public Mapping, py::trampoline_self_life_support {
public:
    using Mapping::Mapping;

    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_forward_impl, &fb, inputs, conditions
        );
    }

    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_inverse_impl, &fb, inputs, conditions
        );
    }
};

class PyFunctionGenerator : public FunctionGenerator, py::trampoline_self_life_support {
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
        .value("int", DataType::dt_int)
        .value("float", DataType::dt_float)
        .value("batch_sizes", DataType::batch_sizes)
        .export_values();

    py::classh<BatchSize>(m, "BatchSize")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        .def_readonly_static("one", &BatchSize::one)
        .def("__str__", &to_string<BatchSize>)
        .def("__repr__", &to_string<BatchSize>);
    m.attr("batch_size") = py::cast(batch_size);

    py::classh<Type>(m, "Type")
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
    m.attr("batch_float") = py::cast(batch_float);
    m.attr("batch_int") = py::cast(batch_int);
    m.attr("batch_four_vec") = py::cast(batch_four_vec);
    m.def("batch_float_array", &batch_float_array, py::arg("count"));
    m.def("batch_four_vec_array", &batch_four_vec_array, py::arg("count"));

    py::classh<InstrCopy>(m, "Instruction")
        .def("__str__", [](const InstrCopy& instr) { return instr.name; } )
        .def_readonly("name", &InstrCopy::name)
        .def_readonly("opcode", &InstrCopy::opcode);

    py::classh<Value>(m, "Value")
        .def(py::init<me_int_t>(), py::arg("value"))
        .def(py::init<double>(), py::arg("value"))
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type)
        .def_readonly("literal_value", &Value::literal_value)
        .def_readonly("local_index", &Value::local_index);
    py::implicitly_convertible<me_int_t, Value>();
    py::implicitly_convertible<double, Value>();
    py::implicitly_convertible<std::string, Value>();

    py::classh<InstructionCall>(m, "InstructionCall")
        .def("__str__", &to_string<InstructionCall>)
        .def("__repr__", &to_string<InstructionCall>)
        .def_property_readonly("instruction", [](const InstructionCall& call) -> InstrCopy {
            return call.instruction;
        })
        .def_readonly("inputs", &InstructionCall::inputs)
        .def_readonly("outputs", &InstructionCall::outputs);

    py::classh<Function>(m, "Function", py::dynamic_attr())
        .def("__str__", &to_string<Function>)
        .def("__repr__", &to_string<Function>)
        .def("store", &Function::store, py::arg("file"))
        .def_static("load", &Function::load, py::arg("file"))
        .def_property_readonly("inputs", &Function::inputs)
        .def_property_readonly("outputs", &Function::outputs)
        .def_property_readonly("locals", &Function::locals)
        .def_property_readonly("globals", &Function::globals)
        .def_property_readonly("instructions", &Function::instructions);

    py::classh<Device> device(m, "Device");
    m.def("cpu_device", &cpu_device, py::return_value_policy::reference);
    m.def("cuda_device", &cuda_device, py::return_value_policy::reference);

    py::classh<MatrixElementApi>(m, "MatrixElementApi")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("file"), py::arg("param_card"))
        .def("on_gpu", &MatrixElementApi::on_gpu)
        .def("particle_count", &MatrixElementApi::particle_count)
        .def("diagram_count", &MatrixElementApi::diagram_count)
        .def("helicity_count", &MatrixElementApi::helicity_count);

    py::classh<Tensor>(m, "Tensor", py::dynamic_attr())
        .def("__dlpack__", &tensor_to_dlpack,
             py::arg("stream")=std::nullopt,
             py::arg("max_version")=std::nullopt,
             py::arg("dl_device")=std::nullopt,
             py::arg("copy")=std::nullopt)
        .def("__dlpack_device__", &dlpack_device);

    py::classh<Context>(m, "Context")
        .def(py::init<>())
        .def(py::init<DevicePtr>(), py::arg("device"))
        .def("load_matrix_element", &Context::load_matrix_element,
             py::arg("file"), py::arg("param_card"))
        .def("define_global", &Context::define_global,
             py::arg("name"), py::arg("dtype"), py::arg("shape"),
             py::arg("requires_grad")=false)
        .def("get_global", &Context::global, py::arg("name"))
        .def("global_requires_grad", &Context::global_requires_grad, py::arg("name"))
        .def("global_exists", &Context::global_exists, py::arg("name"))
        .def("matrix_element", &Context::matrix_element,
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("save", &Context::save, py::arg("file"))
        .def("load", &Context::load, py::arg("file"))
        .def("device", &Context::device);
    m.def("default_context", &default_context);
    m.def("default_cuda_context", &default_cuda_context);

    py::classh<FunctionRuntime>(m, "FunctionRuntime", py::dynamic_attr())
        .def(py::init<Function>(), py::arg("function"))
        .def(py::init<Function, ContextPtr>(), py::arg("function"), py::arg("context"))
        .def("call", &FunctionRuntime::call)
        .def("call_with_grad", &FunctionRuntime::call_with_grad)
        .def("call_backward", &FunctionRuntime::call_backward);

    auto& fb = py::classh<FunctionBuilder>(m, "FunctionBuilder")
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
        .def("product", &FunctionBuilder::product, py::arg("values"))
        .def("function", &FunctionBuilder::function);
    add_instructions(fb);

    py::classh<Mapping, PyMapping>(m, "Mapping", py::dynamic_attr())
        .def(py::init<const std::string&, const TypeVec&, const TypeVec&, const TypeVec&>(),
             py::arg("name"), py::arg("input_types"),
             py::arg("output_types"), py::arg("condition_types"))
        .def("forward_function", &Mapping::forward_function)
        .def("inverse_function", &Mapping::inverse_function)
        .def("build_forward", &Mapping::build_forward,
             py::arg("builder"), py::arg("inputs"), py::arg("conditions"))
        .def("build_inverse", &Mapping::build_inverse,
             py::arg("builder"), py::arg("inputs"), py::arg("conditions"));
    py::classh<Invariant, Mapping>(m, "Invariant")
        .def(py::init<double, double, double>(),
             py::arg("power")=0., py::arg("mass")=0., py::arg("width")=0.);
    py::classh<Luminosity, Mapping>(m, "Luminosity")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("s_lab"), py::arg("s_hat_min"), py::arg("s_hat_max")=0.,
             py::arg("invariant_power")=0., py::arg("mass")=0., py::arg("width")=0.);
    py::classh<TwoParticleDecay, Mapping>(m, "TwoParticleDecay")
        .def(py::init<bool>(), py::arg("com"));
    py::classh<TwoParticleScattering, Mapping>(m, "TwoParticleScattering")
        .def(py::init<bool, double, double, double>(),
             py::arg("com"), py::arg("invariant_power")=0.,
             py::arg("mass")=0., py::arg("width")=0.);
    py::classh<Propagator>(m, "Propagator")
        .def(py::init<double, double, int>(),
             py::arg("mass")=0., py::arg("width")=0., py::arg("integration_order")=0)
        .def_readonly("mass", &Propagator::mass)
        .def_readonly("width", &Propagator::width)
        .def_readonly("integration_order", &Propagator::integration_order);
    py::classh<TPropagatorMapping, Mapping>(m, "TPropagatorMapping")
        .def(py::init<std::vector<std::size_t>, double>(),
             py::arg("integration_order"), py::arg("invariant_power")=0.);
    py::classh<VegasMapping, Mapping>(m, "VegasMapping")
        .def(py::init<std::size_t, std::size_t, const std::string&>(),
             py::arg("dimension"), py::arg("bin_count"), py::arg("prefix")="")
        .def("grid_name", &VegasMapping::grid_name)
        .def("initialize_globals", &VegasMapping::initialize_globals, py::arg("context"));

    py::classh<FastRamboMapping, Mapping>(m, "FastRamboMapping")
        .def(py::init<std::size_t, bool>(), py::arg("n_particles"), py::arg("massless"));

    py::classh<MultiChannelMapping, Mapping>(m, "MultiChannelMapping")
        .def(py::init<std::vector<std::shared_ptr<Mapping>>&>(), py::arg("mappings"));

    py::classh<FunctionGenerator, PyFunctionGenerator>(
             m, "FunctionGenerator", py::dynamic_attr())
        .def(py::init<const std::string&, const TypeVec&, const TypeVec&>(),
             py::arg("name"), py::arg("arg_types"), py::arg("return_types"))
        .def("function", &FunctionGenerator::function)
        .def("build_function", &FunctionGenerator::build_function,
             py::arg("builder"), py::arg("args"));

    auto cuts = py::classh<Cuts, FunctionGenerator>(m, "Cuts");
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
    py::classh<Cuts::CutItem>(m, "CutItem")
        .def(py::init<Cuts::CutObservable, Cuts::LimitType, double, Cuts::PidVec>(),
             py::arg("observable"), py::arg("limit_type"),
             py::arg("value"), py::arg("pids"))
        .def_readonly("observable", &Cuts::CutItem::observable)
        .def_readonly("limit_type", &Cuts::CutItem::limit_type)
        .def_readonly("value", &Cuts::CutItem::value)
        .def_readonly("pids", &Cuts::CutItem::pids);
    cuts.def(py::init<std::vector<int>, std::vector<Cuts::CutItem>>(),
             py::arg("pids"), py::arg("cut_data"))
        .def("sqrt_s_min", &Cuts::sqrt_s_min)
        .def("eta_max", &Cuts::eta_max)
        .def("pt_min", &Cuts::pt_min)
        .def_readonly_static("jet_pids", &Cuts::jet_pids)
        .def_readonly_static("bottom_pids", &Cuts::bottom_pids)
        .def_readonly_static("lepton_pids", &Cuts::lepton_pids)
        .def_readonly_static("missing_pids", &Cuts::missing_pids)
        .def_readonly_static("photon_pids", &Cuts::photon_pids);

    py::classh<Diagram::LineRef>(m, "LineRef")
        .def(py::init<std::string>(), py::arg("str"))
        .def("__repr__", &to_string<Diagram::LineRef>);
    py::implicitly_convertible<std::string, Diagram::LineRef>();
    py::classh<Diagram>(m, "Diagram")
        .def(py::init<std::vector<double>&,
                      std::vector<double>&,
                      std::vector<Propagator>&,
                      std::vector<Diagram::Vertex>&>(),
             py::arg("incoming_masses"),
             py::arg("outgoing_masses"),
             py::arg("propagators"),
             py::arg("vertices"))
        .def_property_readonly("incoming_masses", &Diagram::incoming_masses)
        .def_property_readonly("outgoing_masses", &Diagram::outgoing_masses)
        .def_property_readonly("propagators", &Diagram::propagators)
        .def_property_readonly("vertices", &Diagram::vertices)
        .def_property_readonly("incoming_vertices", &Diagram::incoming_vertices)
        .def_property_readonly("outgoing_vertices", &Diagram::outgoing_vertices)
        .def_property_readonly("propagator_vertices", &Diagram::propagator_vertices);
    py::classh<Topology::Decay>(m, "Decay")
        .def_readonly("index", &Topology::Decay::index)
        .def_readonly("parent_index", &Topology::Decay::parent_index)
        .def_readonly("child_indices", &Topology::Decay::child_indices)
        .def_readonly("mass", &Topology::Decay::mass)
        .def_readonly("width", &Topology::Decay::width);
    auto& topology = py::classh<Topology>(m, "Topology")
        .def(py::init<const Diagram&>(), py::arg("diagram"))
        .def_property_readonly("t_propagator_count", &Topology::t_propagator_count)
        .def_property_readonly("t_integration_order", &Topology::t_integration_order)
        .def_property_readonly("t_propagator_masses", &Topology::t_propagator_masses)
        .def_property_readonly("t_propagator_widths", &Topology::t_propagator_widths)
        .def_property_readonly("decays", &Topology::decays)
        .def_property_readonly("decay_integration_order", &Topology::decay_integration_order)
        .def_property_readonly("outgoing_indices", &Topology::outgoing_indices)
        .def_property_readonly("incoming_masses", &Topology::incoming_masses)
        .def_property_readonly("outgoing_masses", &Topology::outgoing_masses)
        .def("propagator_momentum_terms", &Topology::propagator_momentum_terms);
    py::classh<PhaseSpaceMapping, Mapping> psmap(m, "PhaseSpaceMapping");
    py::enum_<PhaseSpaceMapping::TChannelMode>(psmap, "TChannelMode")
        .value("propagator", PhaseSpaceMapping::propagator)
        .value("rambo", PhaseSpaceMapping::rambo)
        .value("chili", PhaseSpaceMapping::chili)
        .export_values();
    psmap
        .def(py::init<const Topology&, double, bool, double,
                      PhaseSpaceMapping::TChannelMode, const std::optional<Cuts>&,
                      const std::vector<std::vector<std::size_t>>&>(),
             py::arg("topology"), py::arg("cm_energy"),
             py::arg("leptonic")=false, py::arg("invariant_power")=0.8,
             py::arg("t_channel_mode")=PhaseSpaceMapping::propagator,
             py::arg("cuts")=std::nullopt,
             py::arg("permutations")=std::vector<Topology>{})
        .def(py::init<const std::vector<double>&, double, bool, double,
                      PhaseSpaceMapping::TChannelMode, std::optional<Cuts>>(),
             py::arg("masses"), py::arg("cm_energy"),
             py::arg("leptonic")=false, py::arg("invariant_power")=0.8,
             py::arg("mode")=PhaseSpaceMapping::rambo,
             py::arg("cuts")=std::nullopt)
        .def("random_dim", &PhaseSpaceMapping::random_dim)
        .def("particle_count", &PhaseSpaceMapping::particle_count)
        .def("channel_count", &PhaseSpaceMapping::channel_count);


    py::classh<MultiChannelFunction, FunctionGenerator>(m, "MultiChannelFunction")
        .def(py::init<std::vector<std::shared_ptr<FunctionGenerator>>&>(),
             py::arg("functions"));

    py::classh<MatrixElement, FunctionGenerator>(m, "MatrixElement")
        .def(py::init<std::size_t, std::size_t, bool, std::size_t,
                      const std::vector<me_int_t>&>(),
             py::arg("matrix_element_index"), py::arg("particle_count"),
             py::arg("simple_matrix_element")=true, py::arg("channel_count")=1,
             py::arg("amp2_remap")=std::vector<me_int_t>{})
        .def("channel_count", &MatrixElement::channel_count)
        .def("particle_count", &MatrixElement::particle_count);

    py::classh<MLP, FunctionGenerator> mlp(m, "MLP");
    py::enum_<MLP::Activation>(mlp, "Activation")
        .value("relu", MLP::relu)
        .value("leaky_relu", MLP::leaky_relu)
        .value("elu", MLP::elu)
        .value("gelu", MLP::gelu)
        .value("sigmoid", MLP::sigmoid)
        .value("softplus", MLP::softplus)
        .value("linear", MLP::linear)
        .export_values();
    mlp.def(py::init<std::size_t, std::size_t, std::size_t, std::size_t,
                      MLP::Activation, const std::string&>(),
            py::arg("input_dim"),
            py::arg("output_dim"),
            py::arg("hidden_dim") = 32,
            py::arg("layers") = 3,
            py::arg("activation") = MLP::leaky_relu,
            py::arg("prefix") = "")
        .def("input_dim", &MLP::input_dim)
        .def("output_dim", &MLP::output_dim)
        .def("initialize_globals", &MLP::initialize_globals, py::arg("context"));

    py::classh<Flow, Mapping>(m, "Flow")
        .def(py::init<std::size_t, std::size_t, const std::string&, std::size_t, std::size_t,
                      std::size_t, MLP::Activation, bool>(),
             py::arg("input_dim"),
             py::arg("condition_dim") = 0,
             py::arg("prefix") = "",
             py::arg("bin_count") = 10,
             py::arg("subnet_hidden_dim") = 32,
             py::arg("subnet_layers") = 3,
             py::arg("subnet_activation") = MLP::leaky_relu,
             py::arg("invert_spline") = true)
        .def("input_dim", &Flow::input_dim)
        .def("condition_dim", &Flow::condition_dim)
        .def("initialize_globals", &Flow::initialize_globals, py::arg("context"))
        .def("initialize_from_vegas", &Flow::initialize_from_vegas,
             py::arg("context"), py::arg("grid_name"));

    py::classh<PropagatorChannelWeights, FunctionGenerator>(m, "PropagatorChannelWeights")
        .def(py::init<const std::vector<Topology>&,
                      const std::vector<std::vector<std::vector<std::size_t>>>&,
                      const std::vector<std::vector<std::size_t>>>(),
             py::arg("topologies"), py::arg("permutations"), py::arg("channel_indices"));

    py::classh<MomentumPreprocessing, FunctionGenerator>(m, "MomentumPreprocessing")
        .def(py::init<std::size_t>(), py::arg("particle_count"))
        .def("output_dim", &MomentumPreprocessing::output_dim);

    py::classh<ChannelWeightNetwork, FunctionGenerator>(m, "ChannelWeightNetwork")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t,
                      MLP::Activation, const std::string&>(),
             py::arg("channel_count"),
             py::arg("particle_count"),
             py::arg("hidden_dim") = 32,
             py::arg("layers") = 3,
             py::arg("activation") = MLP::leaky_relu,
             py::arg("prefix") = "")
        .def("mlp", &ChannelWeightNetwork::mlp)
        .def("preprocessing", &ChannelWeightNetwork::preprocessing)
        .def("mask_name", &ChannelWeightNetwork::mask_name)
        .def("initialize_globals", &ChannelWeightNetwork::initialize_globals,
             py::arg("context"));

    py::classh<DiscreteSampler, Mapping>(m, "DiscreteSampler")
        .def(py::init<const std::vector<std::size_t>&, const std::string&,
                      const std::vector<std::size_t>&>(),
             py::arg("option_counts"), py::arg("prefix") = "",
             py::arg("dims_with_prior") = std::vector<std::size_t>{})
        .def("option_counts", &DiscreteSampler::option_counts)
        .def("prob_names", &DiscreteSampler::prob_names)
        .def("initialize_globals", &DiscreteSampler::initialize_globals,
             py::arg("context"));

    py::classh<DiscreteFlow, Mapping>(m, "DiscreteFlow")
        .def(py::init<const std::vector<std::size_t>&, const std::string&,
                      const std::vector<std::size_t>&, std::size_t, std::size_t,
                      std::size_t, MLP::Activation>(),
             py::arg("option_counts"),
             py::arg("prefix") = "",
             py::arg("dims_with_prior") = std::vector<std::size_t>{},
             py::arg("condition_dim") = 0,
             py::arg("subnet_hidden_dim") = 32,
             py::arg("subnet_layers") = 3,
             py::arg("subnet_activation") = MLP::leaky_relu)
        .def("option_counts", &DiscreteFlow::option_counts)
        .def("condition_dim", &DiscreteFlow::condition_dim)
        .def("initialize_globals", &DiscreteFlow::initialize_globals,
             py::arg("context"));

    py::classh<VegasGridOptimizer>(m, "VegasGridOptimizer")
        .def("add_data", [](VegasGridOptimizer& opt, py::object values, py::object counts) {
                opt.add_data(
                    dlpack_to_tensor(values, batch_float, 0),
                    dlpack_to_tensor(counts, batch_float_array(opt.input_dim()), 1)
                );
             }, py::arg("values"), py::arg("counts"))
        .def("optimize", &VegasGridOptimizer::optimize)
        .def(py::init<ContextPtr, const std::string&, double>(),
             py::arg("context"), py::arg("grid_name"), py::arg("damping"));

    py::classh<DiscreteOptimizer>(m, "DiscreteOptimizer")
        .def("add_data",
             [](DiscreteOptimizer& opt, std::vector<py::object> values_and_counts) {
                TensorVec input_tensors;
                for (std::size_t i = 1; auto& input : values_and_counts) {
                    input_tensors.push_back(dlpack_to_tensor(
                        input, i % 2 == 0 ? batch_int : batch_float, i
                    ));
                    ++i;
                }
                opt.add_data(input_tensors);
             }, py::arg("values_and_counts"))
        .def("optimize", &DiscreteOptimizer::optimize)
        .def(py::init<ContextPtr, const std::vector<std::string>&>(),
             py::arg("context"), py::arg("prob_names"));

    py::classh<PdfGrid>(m, "PdfGrid")
        .def(py::init<const std::string&>(), py::arg("file"))
        .def_readonly("x", &PdfGrid::x)
        .def_readonly("logx", &PdfGrid::logx)
        .def_readonly("q", &PdfGrid::q)
        .def_readonly("logq2", &PdfGrid::logq2)
        .def_readonly("pids", &PdfGrid::pids)
        .def_readonly("values", &PdfGrid::values)
        .def_readonly("region_sizes", &PdfGrid::region_sizes)
        .def_property_readonly("grid_point_count", &PdfGrid::grid_point_count)
        .def_property_readonly("q_count", &PdfGrid::q_count)
        .def("coefficients_shape", &PdfGrid::coefficients_shape, py::arg("batch_dim")=false)
        .def("logx_shape", &PdfGrid::logx_shape, py::arg("batch_dim")=false)
        .def("logq2_shape", &PdfGrid::logq2_shape, py::arg("batch_dim")=false)
        .def("initialize_globals", &PdfGrid::initialize_globals,
             py::arg("context"), py::arg("prefix")="");

    py::classh<PartonDensity, FunctionGenerator>(m, "PartonDensity")
        .def(py::init<const PdfGrid&, const std::vector<int>&, bool, const std::string&>(),
             py::arg("grid"), py::arg("pids"), py::arg("dynamic_pid")=false,
             py::arg("prefix")="");

    py::classh<AlphaSGrid>(m, "AlphaSGrid")
        .def(py::init<const std::string&>(), py::arg("file"))
        .def_readonly("q", &AlphaSGrid::q)
        .def_readonly("logq2", &AlphaSGrid::logq2)
        .def_readonly("values", &AlphaSGrid::values)
        .def_readonly("region_sizes", &AlphaSGrid::region_sizes)
        .def_property_readonly("q_count", &AlphaSGrid::q_count)
        .def("coefficients_shape", &AlphaSGrid::coefficients_shape,
             py::arg("batch_dim")=false)
        .def("logq2_shape", &AlphaSGrid::logq2_shape, py::arg("batch_dim")=false)
        .def("initialize_globals", &AlphaSGrid::initialize_globals,
             py::arg("context"), py::arg("prefix")="");

    py::classh<RunningCoupling, FunctionGenerator>(m, "RunningCoupling")
        .def(py::init<const AlphaSGrid&, const std::string&>(),
             py::arg("grid"), py::arg("prefix")="");

    py::classh<EnergyScale, FunctionGenerator> scale(m, "EnergyScale");
    py::enum_<EnergyScale::DynamicalScaleType>(scale, "DynamicalScaleType")
        .value("transverse_energy", EnergyScale::transverse_energy)
        .value("transverse_mass", EnergyScale::transverse_mass)
        .value("half_transverse_mass", EnergyScale::half_transverse_mass)
        .value("partonic_energy", EnergyScale::partonic_energy)
        .export_values();
    scale
        .def(py::init<std::size_t>(), py::arg("particle_count"))
        .def(py::init<std::size_t, EnergyScale::DynamicalScaleType>(),
             py::arg("particle_count"), py::arg("type"))
        .def(py::init<std::size_t, double>(),
             py::arg("particle_count"), py::arg("fixed_scale"))
        .def(py::init<std::size_t, EnergyScale::DynamicalScaleType, bool, bool,
                      double, double, double>(),
             py::arg("particle_count"), py::arg("dynamical_scale_type"), py::arg("ren_scale_fixed"),
             py::arg("fact_scale_fixed"), py::arg("ren_scale"), py::arg("fact_scale1"),
             py::arg("fact_scale2"));

    py::classh<DifferentialCrossSection, FunctionGenerator>(m, "DifferentialCrossSection")
        .def(py::init<const std::vector<std::vector<me_int_t>>&, std::size_t,
                      const RunningCoupling&, const std::optional<PdfGrid>&, double,
                      const EnergyScale&, bool, std::size_t,
                      const std::vector<me_int_t>&, bool>(),
             py::arg("pid_options"),
             py::arg("matrix_element_index"),
             py::arg("running_coupling"),
             py::arg("pdf_grid"),
             py::arg("cm_energy"),
             py::arg("energy_scale"),
             py::arg("simple_matrix_element")=true,
             py::arg("channel_count")=1,
             py::arg("amp2_remap")=std::vector<me_int_t>{},
             py::arg("has_mirror")=false)
        .def("pid_options", &DifferentialCrossSection::pid_options);

    py::classh<Unweighter, FunctionGenerator>(m, "Unweighter")
        .def(py::init<const TypeVec&>(), py::arg("types"));
    py::classh<Integrand, FunctionGenerator>(m, "Integrand")
        .def(py::init<const PhaseSpaceMapping&,
                      const DifferentialCrossSection&,
                      const Integrand::AdaptiveMapping&,
                      const Integrand::AdaptiveDiscrete&,
                      const Integrand::AdaptiveDiscrete&,
                      const std::optional<PdfGrid>&,
                      const std::optional<EnergyScale>&,
                      const std::optional<PropagatorChannelWeights>&,
                      const std::optional<ChannelWeightNetwork>&,
                      int,
                      const std::vector<std::size_t>&,
                      const std::vector<std::size_t>&>(),
             py::arg("mapping"),
             py::arg("diff_xs"),
             py::arg("adaptive_map")=std::monostate{},
             py::arg("discrete_before")=std::monostate{},
             py::arg("discrete_after")=std::monostate{},
             py::arg("pdf_grid")=std::nullopt,
             py::arg("energy_scale")=std::nullopt,
             py::arg("prop_chan_weights")=std::nullopt,
             py::arg("chan_weight_net")=std::nullopt,
             py::arg("flags")=0,
             py::arg("channel_indices")=std::vector<std::size_t>{},
             py::arg("active_flavors")=std::vector<std::size_t>{})
        .def("particle_count", &Integrand::particle_count)
        .def("flags", &Integrand::flags)
        .def("vegas_grid_name", &Integrand::vegas_grid_name)
        .def("mapping", &Integrand::mapping)
        .def("diff_xs", &Integrand::diff_xs)
        .def("adaptive_map", &Integrand::adaptive_map)
        .def("discrete_before", &Integrand::discrete_before)
        .def("discrete_after", &Integrand::discrete_after)
        .def("energy_scale", &Integrand::energy_scale)
        .def("prop_chan_weights", &Integrand::prop_chan_weights)
        .def("chan_weight_net", &Integrand::chan_weight_net)
        .def("random_dim", &Integrand::random_dim)
        .def("latent_dims", &Integrand::latent_dims)
        .def_readonly_static("sample", &Integrand::sample)
        .def_readonly_static("unweight", &Integrand::unweight)
        .def_readonly_static("return_momenta", &Integrand::return_momenta)
        .def_readonly_static("return_x1_x2", &Integrand::return_x1_x2)
        .def_readonly_static("return_random", &Integrand::return_random)
        .def_readonly_static("return_latent", &Integrand::return_latent)
        .def_readonly_static("return_channel", &Integrand::return_channel)
        .def_readonly_static("return_chan_weights", &Integrand::return_chan_weights)
        .def_readonly_static("return_cwnet_input", &Integrand::return_cwnet_input)
        .def_readonly_static("return_discrete", &Integrand::return_discrete)
        .def_readonly_static("return_discrete_latent", &Integrand::return_discrete_latent);
    py::classh<IntegrandProbability, FunctionGenerator>(m, "IntegrandProbability")
        .def(py::init<const Integrand&>(), py::arg("integrand"));

    py::classh<EventGenerator::Config>(m, "EventGeneratorConfig")
        .def(py::init<>())
        .def_readwrite("target_count", &EventGenerator::Config::target_count)
        .def_readwrite("vegas_damping", &EventGenerator::Config::vegas_damping)
        .def_readwrite("max_overweight_truncation",
                       &EventGenerator::Config::max_overweight_truncation)
        .def_readwrite("freeze_max_weight_after",
                       &EventGenerator::Config::freeze_max_weight_after)
        .def_readwrite("start_batch_size", &EventGenerator::Config::start_batch_size)
        .def_readwrite("max_batch_size", &EventGenerator::Config::max_batch_size)
        .def_readwrite("survey_min_iters", &EventGenerator::Config::survey_min_iters)
        .def_readwrite("survey_max_iters", &EventGenerator::Config::survey_max_iters)
        .def_readwrite("survey_target_precision",
                       &EventGenerator::Config::survey_target_precision)
        .def_readwrite("optimization_patience",
                       &EventGenerator::Config::optimization_patience)
        .def_readwrite("optimization_threshold",
                       &EventGenerator::Config::optimization_threshold)
        .def_readwrite("batch_size", &EventGenerator::Config::batch_size);
    py::classh<EventGenerator::Status>(m, "EventGeneratorStatus")
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
    py::classh<EventGenerator>(m, "EventGenerator")
        .def_readonly_static("default_config", &EventGenerator::default_config)
        .def(py::init<ContextPtr, const std::vector<Integrand>&,
                      const std::string&, const EventGenerator::Config&,
                      const std::optional<std::string>&>(),
             py::arg("context"), py::arg("channels"), py::arg("file_name"),
             py::arg_v("default_config", EventGenerator::default_config,
                       "EventGenerator.default_config"),
             py::arg("temp_file_dir")=std::nullopt)
        .def("survey", &EventGenerator::survey)
        .def("generate", &EventGenerator::generate)
        .def("status", &EventGenerator::status)
        .def("channel_status", &EventGenerator::channel_status)
        .def_readonly_static("integrand_flags", &EventGenerator::integrand_flags);

    m.def("set_thread_count", [](std::size_t new_count) {
        default_thread_pool().set_thread_count(new_count);
    }, py::arg("new_count"));
    m.def("format_si_prefix", &format_si_prefix, py::arg("value"));
    m.def("format_with_error", &format_with_error, py::arg("value"), py::arg("error"));
    m.def("format_progress", &format_progress, py::arg("progress"), py::arg("width"));
    m.def("initialize_vegas_grid", &initialize_vegas_grid,
          py::arg("context"), py::arg("grid_name"));
    m.def("set_lib_path", &set_lib_path, py::arg("lib_path"));
    m.def("set_simd_vector_size", &set_simd_vector_size, py::arg("vector_size"));

    EventGenerator::set_abort_check_function([]{
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    });
}
