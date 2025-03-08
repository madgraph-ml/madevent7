#include "madevent/phasespace/invariants.h"

using namespace madevent;

Mapping::Result Invariant::build_forward_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto r = inputs[0], s_min = conditions[0], s_max = conditions[1];
    auto [s, det] =
        width != 0 ? fb.breit_wigner_invariant(r, mass, width, s_min, s_max) :
        nu == 0 ? fb.uniform_invariant(r, s_min, s_max) :
        nu == 1 ? fb.stable_invariant(r, mass, s_min, s_max) :
        fb.stable_invariant_nu(r, mass, nu, s_min, s_max);
    return {{s}, det};
}

Mapping::Result Invariant::build_inverse_impl(
    FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
) const {
    auto s = inputs[0], s_min = conditions[0], s_max = conditions[1];
    auto [r, det] =
        width != 0 ? fb.breit_wigner_invariant_inverse(s, mass, width, s_min, s_max) :
        nu == 0 ? fb.uniform_invariant_inverse(s, s_min, s_max) :
        nu == 1 ? fb.stable_invariant_inverse(s, mass, s_min, s_max) :
        fb.stable_invariant_nu_inverse(s, mass, nu, s_min, s_max);
    return {{r}, det};
}
