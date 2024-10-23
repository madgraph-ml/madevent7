// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "madevent/madcode.h"

namespace py = pybind11;
using madevent::FunctionBuilder;

namespace {

void add_instructions(py::class_<FunctionBuilder>& fb) {
    fb.def("stack", &FunctionBuilder::stack, py::arg("args"));
    fb.def("unstack", &FunctionBuilder::unstack, py::arg("in"));
    fb.def("batch_cat", &FunctionBuilder::batch_cat, py::arg("args"));
    fb.def("batch_split", &FunctionBuilder::batch_split, py::arg("in"), py::arg("counts"));
    fb.def("add", &FunctionBuilder::add, py::arg("in1"), py::arg("in2"));
    fb.def("sub", &FunctionBuilder::sub, py::arg("in1"), py::arg("in2"));
    fb.def("mul", &FunctionBuilder::mul, py::arg("in1"), py::arg("in2"));
    fb.def("mul_scalar", &FunctionBuilder::mul_scalar, py::arg("in1"), py::arg("in2"));
    fb.def("clip_min", &FunctionBuilder::clip_min, py::arg("x"), py::arg("min"));
    fb.def("sqrt", &FunctionBuilder::sqrt, py::arg("in"));
    fb.def("square", &FunctionBuilder::square, py::arg("in"));
    fb.def("uniform_phi", &FunctionBuilder::uniform_phi, py::arg("in"));
    fb.def("uniform_phi_inverse", &FunctionBuilder::uniform_phi_inverse, py::arg("in"));
    fb.def("uniform_costheta", &FunctionBuilder::uniform_costheta, py::arg("in"));
    fb.def("uniform_costheta_inverse", &FunctionBuilder::uniform_costheta_inverse, py::arg("in"));
    fb.def("rotate_zy", &FunctionBuilder::rotate_zy, py::arg("p"), py::arg("phi"), py::arg("cos_theta"));
    fb.def("rotate_zy_inverse", &FunctionBuilder::rotate_zy_inverse, py::arg("p"), py::arg("phi"), py::arg("cos_theta"));
    fb.def("boost", &FunctionBuilder::boost, py::arg("p1"), py::arg("p2"));
    fb.def("boost_inverse", &FunctionBuilder::boost_inverse, py::arg("p1"), py::arg("p2"));
    fb.def("boost_beam", &FunctionBuilder::boost_beam, py::arg("p1"), py::arg("rap"));
    fb.def("boost_beam_inverse", &FunctionBuilder::boost_beam_inverse, py::arg("p1"), py::arg("rap"));
    fb.def("com_momentum", &FunctionBuilder::com_momentum, py::arg("sqrt_s"));
    fb.def("com_p_in", &FunctionBuilder::com_p_in, py::arg("e_cm"));
    fb.def("com_angles", &FunctionBuilder::com_angles, py::arg("p"));
    fb.def("s", &FunctionBuilder::s, py::arg("p"));
    fb.def("sqrt_s", &FunctionBuilder::sqrt_s, py::arg("p"));
    fb.def("s_and_sqrt_s", &FunctionBuilder::s_and_sqrt_s, py::arg("p"));
    fb.def("r_to_x1x2", &FunctionBuilder::r_to_x1x2, py::arg("r"), py::arg("s_hat"), py::arg("s_lab"));
    fb.def("x1x2_to_r", &FunctionBuilder::x1x2_to_r, py::arg("x1"), py::arg("x2"), py::arg("s_lab"));
    fb.def("rapidity", &FunctionBuilder::rapidity, py::arg("x1"), py::arg("x2"));
    fb.def("decay_momentum", &FunctionBuilder::decay_momentum, py::arg("s"), py::arg("sqrt_s"), py::arg("m1"), py::arg("m2"));
    fb.def("invt_min_max", &FunctionBuilder::invt_min_max, py::arg("s"), py::arg("s_in1"), py::arg("s_in2"), py::arg("m1"), py::arg("m2"));
    fb.def("invt_to_costheta", &FunctionBuilder::invt_to_costheta, py::arg("s"), py::arg("s_in1"), py::arg("s_in2"), py::arg("m1"), py::arg("m2"), py::arg("t"));
    fb.def("costheta_to_invt", &FunctionBuilder::costheta_to_invt, py::arg("s"), py::arg("s_in1"), py::arg("s_in2"), py::arg("m1"), py::arg("m2"), py::arg("cos_theta"));
    fb.def("two_particle_density_inverse", &FunctionBuilder::two_particle_density_inverse, py::arg("s"), py::arg("m1"), py::arg("m2"));
    fb.def("tinv_two_particle_density", &FunctionBuilder::tinv_two_particle_density, py::arg("det_t"), py::arg("s"), py::arg("s_in1"), py::arg("s_in2"));
    fb.def("tinv_two_particle_density_inverse", &FunctionBuilder::tinv_two_particle_density_inverse, py::arg("det_t"), py::arg("s"), py::arg("s_in1"), py::arg("s_in2"));
    fb.def("uniform_invariant", &FunctionBuilder::uniform_invariant, py::arg("r"), py::arg("s_min"), py::arg("s_max"));
    fb.def("uniform_invariant_inverse", &FunctionBuilder::uniform_invariant_inverse, py::arg("s"), py::arg("s_min"), py::arg("s_max"));
    fb.def("breit_wigner_invariant", &FunctionBuilder::breit_wigner_invariant, py::arg("r"), py::arg("mass"), py::arg("width"), py::arg("s_min"), py::arg("s_max"));
    fb.def("breit_wigner_invariant_inverse", &FunctionBuilder::breit_wigner_invariant_inverse, py::arg("s"), py::arg("mass"), py::arg("width"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant", &FunctionBuilder::stable_invariant, py::arg("r"), py::arg("mass"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_inverse", &FunctionBuilder::stable_invariant_inverse, py::arg("s"), py::arg("mass"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_nu", &FunctionBuilder::stable_invariant_nu, py::arg("r"), py::arg("mass"), py::arg("nu"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_nu_inverse", &FunctionBuilder::stable_invariant_nu_inverse, py::arg("s"), py::arg("mass"), py::arg("nu"), py::arg("s_min"), py::arg("s_max"));
}
}
