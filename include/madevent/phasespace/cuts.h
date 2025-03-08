#pragma once

#include "madevent/madcode.h"

#include <vector>

namespace madevent {

class Cuts {
public:
    using PidVec = std::vector<int>;
    static const PidVec jet_pids;
    static const PidVec bottom_pids;
    static const PidVec lepton_pids;
    static const PidVec missing_pids;
    static const PidVec photon_pids;
    enum CutObservable {obs_pt, obs_eta, obs_dr, obs_mass, obs_sqrt_s};
    enum LimitType {min, max};
    struct CutItem {
        CutObservable observable;
        LimitType limit_type;
        double value;
        PidVec pids;
    };

    Cuts(std::vector<int> _pids, std::vector<CutItem> _cut_data) :
        pids(_pids), cut_data(_cut_data) {}
    ValueVec build_function(FunctionBuilder& fb, Value sqrt_s, Value momenta) const;
    double get_sqrt_s_min() const;
    std::vector<double> get_eta_max() const;
    std::vector<double> get_pt_min() const;

private:
    std::vector<double> get_limits(
        CutObservable observable, LimitType limit_type, double default_value
    ) const;
    void process_single_cuts(
        CutItem cut, std::vector<double>& limits, bool& has_cuts
    ) const;
    void process_pair_cuts(
        CutItem cut,
        std::vector<int64_t>& indices,
        std::vector<double>& limits,
        bool& has_cuts
    ) const;

    std::vector<int> pids;
    std::vector<CutItem> cut_data;
};

}
