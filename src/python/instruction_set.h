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
    fb.def("pop", &FunctionBuilder::pop, py::arg("in"));
    fb.def("batch_cat", &FunctionBuilder::batch_cat, py::arg("args"));
    fb.def("batch_split", &FunctionBuilder::batch_split, py::arg("in"), py::arg("counts"));
    fb.def("cat", &FunctionBuilder::cat, py::arg("args"));
    fb.def("add", &FunctionBuilder::add, py::arg("in1"), py::arg("in2"));
    fb.def("sub", &FunctionBuilder::sub, py::arg("in1"), py::arg("in2"));
    fb.def("mul", &FunctionBuilder::mul, py::arg("in1"), py::arg("in2"));
    fb.def("product", &FunctionBuilder::product, py::arg("in"));
    fb.def("clip_min", &FunctionBuilder::clip_min, py::arg("x"), py::arg("min"));
    fb.def("sqrt", &FunctionBuilder::sqrt, py::arg("in"));
    fb.def("square", &FunctionBuilder::square, py::arg("in"));
    fb.def("pow", &FunctionBuilder::pow, py::arg("in1"), py::arg("in2"));
    fb.def("uniform_phi", &FunctionBuilder::uniform_phi, py::arg("in"));
    fb.def("uniform_phi_inverse", &FunctionBuilder::uniform_phi_inverse, py::arg("in"));
    fb.def("uniform_costheta", &FunctionBuilder::uniform_costheta, py::arg("in"));
    fb.def("uniform_costheta_inverse", &FunctionBuilder::uniform_costheta_inverse, py::arg("in"));
    fb.def("boost", &FunctionBuilder::boost, py::arg("p1"), py::arg("p2"));
    fb.def("boost_inverse", &FunctionBuilder::boost_inverse, py::arg("p1"), py::arg("p2"));
    fb.def("boost_beam", &FunctionBuilder::boost_beam, py::arg("p1"), py::arg("x1"), py::arg("x2"));
    fb.def("boost_beam_inverse", &FunctionBuilder::boost_beam_inverse, py::arg("p1"), py::arg("x1"), py::arg("x2"));
    fb.def("com_momentum", &FunctionBuilder::com_momentum, py::arg("sqrt_s"));
    fb.def("com_p_in", &FunctionBuilder::com_p_in, py::arg("e_cm"));
    fb.def("r_to_x1x2", &FunctionBuilder::r_to_x1x2, py::arg("r"), py::arg("s_hat"), py::arg("s_lab"));
    fb.def("x1x2_to_r", &FunctionBuilder::x1x2_to_r, py::arg("x1"), py::arg("x2"), py::arg("s_lab"));
    fb.def("diff_cross_section", &FunctionBuilder::diff_cross_section, py::arg("x1"), py::arg("x2"), py::arg("pdf1"), py::arg("pdf2"), py::arg("matrix_element"), py::arg("e_cm2"));
    fb.def("two_particle_decay_com", &FunctionBuilder::two_particle_decay_com, py::arg("r_phi"), py::arg("r_cos_theta"), py::arg("m0"), py::arg("m1"), py::arg("m2"));
    fb.def("two_particle_decay", &FunctionBuilder::two_particle_decay, py::arg("r_phi"), py::arg("r_cos_theta"), py::arg("m0"), py::arg("m1"), py::arg("m2"), py::arg("p0"));
    fb.def("two_particle_scattering_com", &FunctionBuilder::two_particle_scattering_com, py::arg("r_phi"), py::arg("pa"), py::arg("pb"), py::arg("t"), py::arg("m1"), py::arg("m2"));
    fb.def("two_particle_scattering", &FunctionBuilder::two_particle_scattering, py::arg("r_phi"), py::arg("pa"), py::arg("pb"), py::arg("t"), py::arg("m1"), py::arg("m2"));
    fb.def("t_inv_min_max", &FunctionBuilder::t_inv_min_max, py::arg("pa"), py::arg("pb"), py::arg("m1"), py::arg("m2"));
    fb.def("invariants_from_momenta", &FunctionBuilder::invariants_from_momenta, py::arg("p_ext"), py::arg("factors"));
    fb.def("sde2_channel_weights", &FunctionBuilder::sde2_channel_weights, py::arg("invariants"), py::arg("masses"), py::arg("widths"), py::arg("indices"));
    fb.def("uniform_invariant", &FunctionBuilder::uniform_invariant, py::arg("r"), py::arg("s_min"), py::arg("s_max"));
    fb.def("uniform_invariant_inverse", &FunctionBuilder::uniform_invariant_inverse, py::arg("s"), py::arg("s_min"), py::arg("s_max"));
    fb.def("breit_wigner_invariant", &FunctionBuilder::breit_wigner_invariant, py::arg("r"), py::arg("mass"), py::arg("width"), py::arg("s_min"), py::arg("s_max"));
    fb.def("breit_wigner_invariant_inverse", &FunctionBuilder::breit_wigner_invariant_inverse, py::arg("s"), py::arg("mass"), py::arg("width"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant", &FunctionBuilder::stable_invariant, py::arg("r"), py::arg("mass"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_inverse", &FunctionBuilder::stable_invariant_inverse, py::arg("s"), py::arg("mass"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_nu", &FunctionBuilder::stable_invariant_nu, py::arg("r"), py::arg("mass"), py::arg("nu"), py::arg("s_min"), py::arg("s_max"));
    fb.def("stable_invariant_nu_inverse", &FunctionBuilder::stable_invariant_nu_inverse, py::arg("s"), py::arg("mass"), py::arg("nu"), py::arg("s_min"), py::arg("s_max"));
    fb.def("fast_rambo_r_to_u", &FunctionBuilder::fast_rambo_r_to_u, py::arg("r"));
    fb.def("rambo_four_vectors_massless", &FunctionBuilder::rambo_four_vectors_massless, py::arg("u"), py::arg("e_cm"), py::arg("cos_theta"), py::arg("phi"));
    fb.def("rambo_four_vectors_massive", &FunctionBuilder::rambo_four_vectors_massive, py::arg("u"), py::arg("e_cm"), py::arg("cos_theta"), py::arg("phi"), py::arg("masses"));
    fb.def("cut_unphysical", &FunctionBuilder::cut_unphysical, py::arg("w_in"), py::arg("p"), py::arg("x1"), py::arg("x2"));
    fb.def("cut_pt", &FunctionBuilder::cut_pt, py::arg("p"), py::arg("min_max"));
    fb.def("cut_eta", &FunctionBuilder::cut_eta, py::arg("p"), py::arg("min_max"));
    fb.def("cut_dr", &FunctionBuilder::cut_dr, py::arg("p"), py::arg("indices"), py::arg("min_max"));
    fb.def("cut_m_inv", &FunctionBuilder::cut_m_inv, py::arg("p"), py::arg("indices"), py::arg("min_max"));
    fb.def("cut_sqrt_s", &FunctionBuilder::cut_sqrt_s, py::arg("p"), py::arg("min_max"));
    fb.def("chili_forward", &FunctionBuilder::chili_forward, py::arg("r"), py::arg("e_cm"), py::arg("m_out"), py::arg("pt_min"), py::arg("y_max"));
    fb.def("matrix_element", &FunctionBuilder::matrix_element, py::arg("momenta"), py::arg("index"));
    fb.def("matrix_element_multichannel", &FunctionBuilder::matrix_element_multichannel, py::arg("momenta"), py::arg("amp2_remap"), py::arg("index"), py::arg("channel_count"));
    fb.def("pdf", &FunctionBuilder::pdf, py::arg("x"), py::arg("q2"), py::arg("pid"));
    fb.def("matmul", &FunctionBuilder::matmul, py::arg("x"), py::arg("weight"), py::arg("bias"));
    fb.def("leaky_relu", &FunctionBuilder::leaky_relu, py::arg("in"));
    fb.def("rqs_activation", &FunctionBuilder::rqs_activation, py::arg("input"), py::arg("bin_count"));
    fb.def("rqs_find_bin", &FunctionBuilder::rqs_find_bin, py::arg("input"), py::arg("in_sizes"), py::arg("out_sizes"), py::arg("derivatives"));
    fb.def("rqs_forward", &FunctionBuilder::rqs_forward, py::arg("input"), py::arg("condition"));
    fb.def("rqs_inverse", &FunctionBuilder::rqs_inverse, py::arg("input"), py::arg("condition"));
    fb.def("softmax", &FunctionBuilder::softmax, py::arg("input"));
    fb.def("softmax_prior", &FunctionBuilder::softmax_prior, py::arg("input"), py::arg("prior"));
    fb.def("sample_discrete", &FunctionBuilder::sample_discrete, py::arg("r"), py::arg("option_count"));
    fb.def("sample_discrete_probs", &FunctionBuilder::sample_discrete_probs, py::arg("r"), py::arg("probs"));
    fb.def("permute_momenta", &FunctionBuilder::permute_momenta, py::arg("momenta"), py::arg("permutations"), py::arg("index"));
    fb.def("gather", &FunctionBuilder::gather, py::arg("index"), py::arg("choices"));
    fb.def("gather_int", &FunctionBuilder::gather_int, py::arg("index"), py::arg("choices"));
    fb.def("select", &FunctionBuilder::select, py::arg("input"), py::arg("indices"));
    fb.def("one_hot", &FunctionBuilder::one_hot, py::arg("index"), py::arg("option_count"));
    fb.def("nonzero", &FunctionBuilder::nonzero, py::arg("input"));
    fb.def("batch_gather", &FunctionBuilder::batch_gather, py::arg("indices"), py::arg("values"));
    fb.def("scatter", &FunctionBuilder::scatter, py::arg("indices"), py::arg("target"), py::arg("source"));
    fb.def("random", &FunctionBuilder::random, py::arg("batch_size"), py::arg("count"));
    fb.def("unweight", &FunctionBuilder::unweight, py::arg("weights"), py::arg("max_weight"));
    fb.def("vegas_forward", &FunctionBuilder::vegas_forward, py::arg("input"), py::arg("grid"));
    fb.def("vegas_inverse", &FunctionBuilder::vegas_inverse, py::arg("input"), py::arg("grid"));
}
}
