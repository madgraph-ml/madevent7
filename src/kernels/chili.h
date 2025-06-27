#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

template<typename T>
KERNELSPEC void kernel_chili_forward(
    FIn<T,1> r, FIn<T,0> e_cm, FIn<T,1> m_out, FIn<T,1> pt_min, FIn<T,1> y_max,
    FOut<T,2> p_ext, FOut<T,0> x1, FOut<T,0> x2, FOut<T,0> det
) {
    // Handle the first n-1 outgoing t-channel particles
    FVal<T> e_sum(0.), px_sum(0.), py_sum(0.), pz_sum(0.);
    FVal<T> det_tmp(1.);
    FVal<T> _e_cm(e_cm);
    auto e_cm2 = _e_cm * _e_cm;
    auto pt_max = _e_cm / 2; // generally too large, thats why we get x1x2 > 1 and NaNs
    auto pt2_max = pt_max * pt_max;
    auto n_out = m_out.size();
    for (std::size_t i = 0; i < n_out - 1; ++i) {
        FVal<T> pt_min_i(pt_min[i]), y_max_i(y_max[i]), m_out_i(m_out[i]);
        auto p_i = p_ext[i + 2];
        FVal<T> r_pt(r[i]), r_y(r[n_out - 1 + i]), r_phi(r[2 * n_out - 1 + i]);

        // get the pts
        auto pt2_min_i = pt_min_i;

        auto pt2_withcut = 1. / (r_pt / pt2_max + (1. - r_pt) / pt2_min_i);
        auto pt_withcut = sqrt(pt2_withcut);
        auto det_pt_withcut = pt2_withcut * pt2_withcut * (1. / pt2_min_i - 1. / pt2_max);

        auto pt_nocut = 2. * m_out_i * pt_max * r_pt / (2. * m_out_i + pt_max * (1. - r_pt));
        auto pt2_nocut = pt_nocut * pt_nocut;
        auto factor_pt = (2 * m_out_i + pt_nocut);
        auto denom_pt = m_out_i * (2 * m_out_i + pt_max);
        auto det_pt_nocut = pt_nocut * pt_max * factor_pt * factor_pt / denom_pt;

        auto has_pt_cut = pt2_min_i > 1e-9;
        auto pt = where(has_pt_cut, pt_withcut, pt_nocut);
        auto pt2 = where(has_pt_cut, pt2_withcut, pt2_nocut);
        auto det_pt = where(has_pt_cut, det_pt_withcut, det_pt_nocut);

        // get first n-1 rapidities and phi
        auto y_max_calc = log(sqrt(e_cm2 / 4. / pt2) + sqrt(e_cm2 / 4. / pt2 - 1.0));
        y_max_calc = min(y_max_i, y_max_calc);
        auto y = y_max_calc * (2. * r_y - 1.0);
        auto phi = 2. * PI * r_phi + atan2(py_sum, px_sum);

        det_tmp = det_tmp / 4.
            * det_pt
            * 2. * y_max_calc
            * 2. * PI;

        // calculate the momenta
        auto sinhy = sinh(y);
        auto coshy = sqrt(1. + sinhy * sinhy);
        auto mt = sqrt(pt2 + m_out_i * m_out_i);
        auto e_i = mt * coshy;
        auto px_i = pt * cos(phi);
        auto py_i = pt * sin(phi);
        auto pz_i = mt * sinhy;
        p_i[0] = e_i;
        p_i[1] = px_i;
        p_i[2] = py_i;
        p_i[3] = pz_i;
        e_sum = e_sum + e_i;
        px_sum = px_sum + px_i;
        py_sum = py_sum + py_i;
        pz_sum = pz_sum + pz_i;
    }

    // Build last t-channel momentum
    FVal<T> r_y(r[2 * n_out - 2]), y_max_n(y_max[n_out - 1]), m_out_n(m_out[n_out - 1]);
    auto p_n = p_ext[n_out + 1];
    auto mj2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum;
    auto yj = 0.5 * log((e_sum + pz_sum) / (e_sum - pz_sum));
    yj = where(e_sum < EPS, 99.0, yj);
    auto ptj2 = px_sum * px_sum + py_sum * py_sum;
    auto m2 = m_out_n * m_out_n;
    auto qt = sqrt(ptj2 + m2);
    auto mt = sqrt(ptj2 + max(mj2, 0.));
    auto y_min_calc = -log(_e_cm / qt * (1.0 - mt / _e_cm * exp(-yj)));
    y_min_calc = max(y_min_calc, -y_max_n); // apply potential cuts
    auto y_max_calc = log(_e_cm / qt * (1.0 - mt / _e_cm * exp(yj)));
    y_max_calc = min(y_max_calc, y_max_n); // apply potential cuts
    auto dely = y_max_calc - y_min_calc;
    auto yn = y_min_calc + r_y * dely;
    det_tmp = det_tmp * dely / e_cm2;
    auto sinhyn = sinh(yn);
    auto coshyn = sqrt(1. + sinhyn * sinhyn);
    auto e_n = qt * coshyn;
    auto pz_n = qt * sinhyn;
    p_n[0] = e_n;
    p_n[1] = -px_sum;
    p_n[2] = -py_sum;
    p_n[3] = pz_n;
    e_sum = e_sum + e_n;
    pz_sum = pz_sum + pz_n;

    // Build incoming momenta
    auto pp = e_sum + pz_sum;
    auto pm = e_sum - pz_sum;
    auto p_a = p_ext[0], p_b = p_ext[1];
    p_a[0] = pp / 2.;
    p_a[1] = 0.;
    p_a[2] = 0.;
    p_a[3] = pp / 2.;
    p_b[0] = pm / 2.;
    p_b[1] = 0.;
    p_b[2] = 0.;
    p_b[3] = - pm / 2.;

    // Get the bjorken variables
    x1 = pp / e_cm;
    x2 = pm / e_cm;

    // keep invalid point but set det/weight to zero (important to obtain correct integral)
    det = where((x1 < 1.) & (x2 < 1.), det_tmp, 0.);
}

}
}
