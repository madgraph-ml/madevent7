#pragma once

#include "madevent/madcode.h"
#include "madevent/phasespace/base.h"
#include "madevent/util.h"

#include <vector>

namespace madevent {

class Observable : public FunctionGenerator {
public:
    static const std::vector<int> jet_pids;
    static const std::vector<int> bottom_pids;
    static const std::vector<int> lepton_pids;
    static const std::vector<int> missing_pids;
    static const std::vector<int> photon_pids;

    enum class Obs {
        e,
        px,
        py,
        pz,
        mass,
        pt,
        p_mag,
        phi,
        theta,
        y,
        eta,
        delta_eta,
        delta_phi,
        delta_r,
        sqrt_s
    };

    Observable(
        const std::vector<int>& pids,
        Obs observable,
        const nested_vector2<int>& select_pids,
        bool sum_momenta = false,
        bool sum_observable = false,
        const std::optional<Obs>& order_observable = std::nullopt,
        bool ignore_incoming = true
    );

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    Obs _observable;
    Obs _order_observable;
    std::nested_vector2<int> _indices;
    bool _sum_momenta;
    bool _sum_observable;
};

} // namespace madevent
