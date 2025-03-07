#include "madevent/phasespace/cuts.h"

#include <ranges>

using namespace madevent;

namespace {

void update_cuts(double& min_cut, double& max_cut, Cuts::LimitType limit_type, double value) {
    if (limit_type == Cuts::min) {
        if (min_cut < value) {
            min_cut = value;
        }
    } else if (max_cut > value) {
        max_cut = value;
    }
}

}

const Cuts::PidVec Cuts::jet_pids {1, 2, 3, 4, -1, -2, -3, -4, 21};
const Cuts::PidVec Cuts::bottom_pids {-5, 5};
const Cuts::PidVec Cuts::lepton_pids {11, 13, 15, -11, -13, -15};
const Cuts::PidVec Cuts::missing_pids {12, 14, 16, -12, -14, -16};
const Cuts::PidVec Cuts::photon_pids {22};

ValueList Cuts::build_function(
    FunctionBuilder& fb, Value sqrt_s, Value momenta
) const {
    bool has_pt_cuts(false), has_eta_cuts(false), has_dr_cuts(false);
    bool has_mass_cuts(false), has_sqrt_s_cuts(false);
    double inf = std::numeric_limits<double>::infinity();
    int n_out = pids.size();
    std::vector<double> pt_cuts(2 * n_out, 0.), eta_cuts(2 * n_out, 0.);
    std::vector<double> dr_cuts, mass_cuts, sqrt_s_cuts{0., inf};
    std::vector<int64_t> dr_indices, mass_indices;
    for (std::size_t i = n_out; i < 2 * n_out; ++i) {
        pt_cuts.at(i) = inf;
        eta_cuts.at(i) = inf;
    }

    // TODO: reduce code duplication
    for (auto& cut : cut_data) {
        switch (cut.observable) {
        case obs_pt:
            process_single_cuts(cut, pt_cuts, has_pt_cuts);
            break;
        case obs_eta:
            process_single_cuts(cut, eta_cuts, has_eta_cuts);
            break;
        case obs_dr:
            process_pair_cuts(cut, dr_indices, dr_cuts, has_dr_cuts);
            break;
        case obs_mass:
            process_pair_cuts(cut, mass_indices, mass_cuts, has_mass_cuts);
            break;
        case obs_sqrt_s:
            update_cuts(sqrt_s_cuts.at(0), sqrt_s_cuts.at(1), cut.limit_type, cut.value);
            has_sqrt_s_cuts = true;
            break;
        }
    }

    ValueList weights;
    if (has_pt_cuts) {
        weights.push_back(fb.cut_pt(momenta, Value(pt_cuts, {n_out, 2})));
    }
    if (has_eta_cuts) {
        weights.push_back(fb.cut_eta(momenta, Value(eta_cuts, {n_out, 2})));
    }
    if (has_dr_cuts) {
        weights.push_back(fb.cut_dr(
            momenta,
            Value(dr_indices, {static_cast<int>(dr_indices.size()) / 2, 2}),
            Value(dr_cuts, {static_cast<int>(dr_cuts.size()) / 2, 2})
        ));
    }
    if (has_mass_cuts) {
        weights.push_back(fb.cut_m_inv(
            momenta,
            Value(mass_indices, {static_cast<int>(mass_indices.size()) / 2, 2}),
            Value(mass_cuts, {static_cast<int>(mass_cuts.size()) / 2, 2})
        ));
    }
    if (has_sqrt_s_cuts) {
        weights.push_back(fb.cut_sqrt_s(momenta, Value(sqrt_s_cuts, {2})));
    }
    return weights;
}

double Cuts::get_sqrt_s_min() const {
    double sqrt_s_min = 0.;
    for (auto& cut : cut_data) {
        if (
            cut.observable == Cuts::obs_sqrt_s &&
            cut.limit_type == Cuts::min &&
            sqrt_s_min < cut.value
        ) {
            sqrt_s_min = cut.value;
        }
    }
    return sqrt_s_min;
}

std::vector<double> Cuts::get_eta_max() const {
    return get_limits(Cuts::obs_eta, Cuts::max, std::numeric_limits<double>::infinity());
}

std::vector<double> Cuts::get_pt_min() const {
    return get_limits(Cuts::obs_pt, Cuts::min, 0.);
}

std::vector<double> Cuts::get_limits(
    CutObservable observable, LimitType limit_type, double default_value
) const {
    std::vector<double> limits(pids.size(), default_value);
    for (auto& cut : cut_data) {
        if (cut.observable == observable && cut.limit_type == limit_type) {
            for (auto [limit, pid] : std::views::zip(limits, pids)) {
                if (
                    std::find(cut.pids.begin(), cut.pids.end(), pid) != cut.pids.end() &&
                    (
                        (limit_type == Cuts::max && (limit > cut.value)) ||
                        (limit_type == Cuts::min && (limit < cut.value))
                    )
                ) {
                    limit = cut.value;
                }
            }
        }
    }
    return limits;
}

void Cuts::process_pair_cuts(
    CutItem cut, std::vector<int64_t>& indices, std::vector<double>& limits, bool& has_cuts
) const {
    double inf = std::numeric_limits<double>::infinity();
    std::size_t i = 0;
    std::vector<int64_t> indices2;
    std::vector<double> limits2;
    for (auto pid_i : pids) {
        std::size_t j = i + 1;
        for (auto pid_j : pids | std::views::drop(i + 1)) {
            if (
                std::find(cut.pids.begin(), cut.pids.end(), pid_i) != cut.pids.end() &&
                std::find(cut.pids.begin(), cut.pids.end(), pid_j) != cut.pids.end()
            ) {
                indices.push_back(i);
                indices2.push_back(j);
                // TODO: update existing cuts
                if (cut.limit_type == Cuts::min) {
                    limits.push_back(cut.value);
                    limits2.push_back(inf);
                } else {
                    limits.push_back(0.);
                    limits2.push_back(cut.value);
                }
                has_cuts = true;
            }
            ++j;
        }
        ++i;
    }
    indices.insert(indices.end(), indices2.begin(), indices2.end());
    limits.insert(limits.end(), limits2.begin(), limits2.end());
}

void Cuts::process_single_cuts(
    CutItem cut, std::vector<double>& limits, bool& has_cuts
) const {
    std::size_t i = 0;
    for (auto pid : pids) {
        if (std::find(cut.pids.begin(), cut.pids.end(), pid) != cut.pids.end()) {
            auto& limit_min = limits.at(i);
            auto& limit_max = limits.at(pids.size() + i);
            update_cuts(limit_min, limit_max, cut.limit_type, cut.value);
            has_cuts = true;
        }
        ++i;
    }
}
