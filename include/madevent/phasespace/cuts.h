#pragma once

#include "madevent/madcode.h"
#include "madevent/phasespace/base.h"

#include <vector>

namespace madevent {

class Cuts : public FunctionGenerator {
public:
    using PidVec = std::vector<int>;
    static const PidVec jet_pids;
    static const PidVec bottom_pids;
    static const PidVec lepton_pids;
    static const PidVec missing_pids;
    static const PidVec photon_pids;
    enum CutObservable { obs_pt, obs_eta, obs_dr, obs_mass, obs_sqrt_s };
    enum LimitType { min, max };
    struct CutItem {
        CutObservable observable;
        LimitType limit_type;
        double value;
        PidVec pids;
    };

    Cuts(std::vector<int> _pids, std::vector<CutItem> _cut_data);
    double sqrt_s_min() const;
    std::vector<double> eta_max() const;
    std::vector<double> pt_min() const;

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
    std::vector<double>
    limits(CutObservable observable, LimitType limit_type, double default_value) const;
    void
    process_single_cuts(CutItem cut, std::vector<double>& limits, bool& has_cuts) const;
    void process_pair_cuts(
        CutItem cut,
        std::vector<me_int_t>& indices,
        std::vector<double>& limits,
        bool& has_cuts
    ) const;

    std::vector<int> _pids;
    std::vector<CutItem> _cut_data;
};

} // namespace madevent
