#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template<typename T>
struct FourMom {
    KERNELSPEC FVal<T>& operator[](int i) { return p[i]; }
    KERNELSPEC FVal<T>& operator[](int i) const { return p[i]; }
    FVal<T> p[4];
};

template<typename F, typename S>
struct Pair {
    F first;
    S second;
};

template<typename T>
KERNELSPEC FourMom<T> load_mom(FIn<T,1> p) {
    return {p[0], p[1], p[2], p[3]};
}

template<typename T>
KERNELSPEC void store_mom(FOut<T,1> p_to, FourMom<T> p_from) {
    for (int i = 0; i < 4; ++i) p_to[i] = p_from[i];
}

template<typename T>
KERNELSPEC FVal<T> kaellen(FVal<T> x, FVal<T> y, FVal<T> z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

template<typename T>
KERNELSPEC FVal<T> lsquare(FourMom<T> p) {
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
}

template<typename T>
KERNELSPEC FourMom<T> rotate(FourMom<T> p, FourMom<T> q) {
    // this function is based on the rotxxx subroutine from HELAS used in MG5 (aloha_functions.f)
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq = sqrt(qt2 + q[3] * q[3]);
    auto qt = sqrt(qt2);
    return {
        p[0],
        q[1] * q[3] / qq / qt * p[1] - q[2] / qt * p[2] + q[1] / qq * p[3],
        q[2] * q[3] / qq / qt * p[1] + q[1] / qt * p[2] + q[2] / qq * p[3],
        -qt / qq * p[1] + q[3] / qq * p[3]
    };
}

template<typename T>
KERNELSPEC FourMom<T> boost(FourMom<T> k, FourMom<T> p_boost, FVal<T> sign) {
    // Perform the boost
    // This is in fact a numerically more stable implementation than often used
    auto p2_boost = lsquare<T>(p_boost);
    auto rsq = sqrt(where(p2_boost < EPS2, EPS2, p2_boost));
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    return FourMom<T>{
        e,
        k[1] + c1 * p_boost[1],
        k[2] + c1 * p_boost[2],
        k[3] + c1 * p_boost[3]
    };
}

template<typename T>
KERNELSPEC void boost_beam(
    FIn<T,2> q, FVal<T> x1, FVal<T> x2, FVal<T> sign, FOut<T,2> p_out
) {
    auto exp_rap = sqrt(x1 / x2);
    auto exp_rap_inv = 1. / exp_rap;
    auto cosh_rap = 0.5 * (exp_rap + exp_rap_inv);
    auto sinh_rap = 0.5 * (exp_rap - exp_rap_inv);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto q_i = q[i];
        auto p_out_i = p_out[i];
        p_out_i[0] = q_i[0] * cosh_rap + sign * q_i[3] * sinh_rap;
        p_out_i[1] = q_i[1];
        p_out_i[2] = q_i[2];
        p_out_i[3] = q_i[3] * cosh_rap + sign * q_i[0] * sinh_rap;
    }
}

template<typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> two_particle_decay(
    FVal<T> r_phi, FVal<T> r_cos_theta, FVal<T> m0, FVal<T> m1, FVal<T> m2
) {
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto m0_clip = where(m0 >= EPS, m0, EPS);

    // this part is based on the mom2cx subroutine from HELAS used in MG5 (aloha_functions.f)
    auto ed = (m1 - m2) * (m1 + m2) / m0_clip;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + m0 * m0;
    auto pp = 0.5 * where(
        m1 * m2 == 0., m0 - fabs(ed), sqrt(where(pp2 >= EPS, pp2, EPS))
    );
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    auto e1 = 0.5 * (m0 + ed);
    FourMom<T> p1{
        where(e1 > 0., e1, 0.),
        pp * sin_theta * cos(phi),
        pp * sin_theta * sin(phi),
        pp * cos_theta
    };

    auto det = PI * pp / m0_clip;
    return {p1, det};
}

#define NANCHECK(name) if (isnan(name)) std::println("nan: {}", #name);

template<typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> two_particle_scattering(
    FVal<T> r_phi, FourMom<T> pa_com, FVal<T> s_tot, FVal<T> t,
    FVal<T> m1, FVal<T> m2, FVal<T> ma_2, FVal<T> mb_2
) {
    // this function is based on the gentcms subroutine in MG5 (genps.f)
    auto m_tot = sqrt(s_tot);

    auto ed = (m1 - m2) * (m1 + m2) / m_tot;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + s_tot;
    auto pp = 0.5 * where(
        m1 * m2 == 0., m_tot - fabs(ed), sqrt(where(pp2 >= EPS, pp2, EPS))
    );

    auto pa_com_mag = sqrt(
        pa_com[1] * pa_com[1] + pa_com[2] * pa_com[2] + pa_com[3] * pa_com[3]
    );
    FourMom<T> p1_com;
    auto e1_com = 0.5 * (m_tot + ed);
    p1_com[0] = where(e1_com >= 0., e1_com, 0.);
    p1_com[3] = -(m1 * m1 + ma_2 + t - 2. * p1_com[0] * pa_com[0]) / (2.0 * pa_com_mag);
    auto pt2 = pp * pp - p1_com[3] * p1_com[3];
    auto pt = sqrt(where(pt2 > 0., pt2, 0.));
    auto phi = PI * (2. * r_phi - 1.);
    p1_com[1] = pt * cos(phi);
    p1_com[2] = pt * sin(phi);
    //if (p1_com[3] > pp) std::println("FALSCH!!!");

    auto det = PI / (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2)));
    return {p1_com, det};
}

// Kernels

template<typename T>
KERNELSPEC void kernel_boost(FIn<T,1> p1, FIn<T,1> p2, FOut<T,1> p_out) {
    store_mom<T>(p_out, boost<T>(load_mom<T>(p1), load_mom<T>(p2), 1.0));
}

template<typename T>
KERNELSPEC void kernel_boost_inverse(FIn<T,1> p1, FIn<T,1> p2, FOut<T,1> p_out) {
    store_mom<T>(p_out, boost<T>(load_mom<T>(p1), load_mom<T>(p2), -1.0));
}

template<typename T>
KERNELSPEC void kernel_boost_beam(FIn<T,2> p1, FIn<T,0> x1, FIn<T,0> x2, FOut<T,2> p_out) {
    boost_beam<T>(p1, x1, x2, 1.0, p_out);
}

template<typename T>
KERNELSPEC void kernel_boost_beam_inverse(FIn<T,2> p1, FIn<T,0> x1, FIn<T,0> x2, FOut<T,2> p_out) {
    boost_beam<T>(p1, x1, x2, -1.0, p_out);
}

template<typename T>
KERNELSPEC void kernel_com_momentum(FIn<T,0> sqrt_s, FOut<T,1> p) {
    p[0] = sqrt_s;
    p[1] = 0;
    p[2] = 0;
    p[3] = 0;
}

template<typename T>
KERNELSPEC void kernel_com_p_in(FIn<T,0> e_cm, FOut<T,1> p1, FOut<T,1> p2) {
    auto p_com = e_cm / 2;
    p1[0] = p_com;
    p1[1] = 0;
    p1[2] = 0;
    p1[3] = p_com;
    p2[0] = p_com;
    p2[1] = 0;
    p2[2] = 0;
    p2[3] = -p_com;
}

template<typename T>
KERNELSPEC void kernel_r_to_x1x2(
    FIn<T,0> r, FIn<T,0> s_hat, FIn<T,0> s_lab, FOut<T,0> x1, FOut<T,0> x2, FOut<T,0> det
) {
    auto tau = s_hat / s_lab;
    x1 = pow(tau, r);
    x2 = pow(tau, (1 - r));
    det = fabs(log(tau)) / s_lab;
}

template<typename T>
KERNELSPEC void kernel_x1x2_to_r(
    FIn<T,0> x1, FIn<T,0> x2, FIn<T,0> s_lab, FOut<T,0> r, FOut<T,0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

template<typename T>
KERNELSPEC void kernel_diff_cross_section(
    FIn<T,0> x1, FIn<T,0> x2, FIn<T,0> pdf1, FIn<T,0> pdf2,
    FIn<T,0> matrix_element, FIn<T,0> e_cm2,
    FOut<T,0> result
) {
    result = INV_GEV2_TO_PB * matrix_element * pdf1 * pdf2
        / (2. * e_cm2 * x1 * x1 * x2 * x2);
}

template<typename T>
KERNELSPEC void kernel_two_particle_decay_com(
    FIn<T,0> r_phi, FIn<T,0> r_cos_theta, FIn<T,0> m0, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,1> p1, FOut<T,1> p2, FOut<T,0> det
) {
    auto decay_out = two_particle_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, p1_tmp);
    det = det_tmp;
    auto e2 = m0 - p1_tmp[0];
    p2[0] = where(e2 >= 0., e2, 0.);
    p2[1] = -p1_tmp[1];
    p2[2] = -p1_tmp[2];
    p2[3] = -p1_tmp[3];
}

template<typename T>
KERNELSPEC void kernel_two_particle_decay(
    FIn<T,0> r_phi, FIn<T,0> r_cos_theta, FIn<T,0> m0, FIn<T,0> m1, FIn<T,0> m2, FIn<T,1> p0,
    FOut<T,1> p1, FOut<T,1> p2, FOut<T,0> det
) {
    auto decay_out = two_particle_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, boost<T>(p1_tmp, load_mom<T>(p0), 1.));
    det = det_tmp;
    auto e2 = p0[0] - p1[0];
    p2[0] = where(e2 >= 0., e2, 0.);
    p2[1] = p0[1] - p1[1];
    p2[2] = p0[2] - p1[2];
    p2[3] = p0[3] - p1[3];
}

template<typename T>
KERNELSPEC void kernel_two_particle_scattering_com(
    FIn<T,0> r_phi, FIn<T,1> pa, FIn<T,1> pb, FIn<T,0> t, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,1> p1, FOut<T,1> p2, FOut<T,0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) p_tot[i] = pa[i] + pb[i];
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto scatter_out = two_particle_scattering<T>(
        r_phi, load_mom<T>(pa), s_tot, t, m1, m2, ma_2, mb_2
    );
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    store_mom<T>(p1, p1_com);
    for (int i = 0; i < 4; ++i) p2[i] = p_tot[i] - p1_com[i];
    det = det_tmp;
    /*auto s_tot_test = sqrt(s_tot);
    std::println("COM pa = {} ma = {}", load_mom<T>(pa), sqrt(lsquare<T>(load_mom<T>(pa))));
    std::println("COM pb = {} mb = {}", load_mom<T>(pb), sqrt(lsquare<T>(load_mom<T>(pb))));
    std::println("COM p1 = {} m1 = {}", load_mom<T>(p1), sqrt(lsquare<T>(load_mom<T>(p1))));
    std::println("COM p2 = {} m2 = {}", load_mom<T>(p2), sqrt(lsquare<T>(load_mom<T>(p2))));
    auto m1_test = sqrt(lsquare<T>(load_mom<T>(p1)));
    auto m2_test = sqrt(lsquare<T>(load_mom<T>(p2)));
    if (s_tot_test < m1 + m2) __builtin_trap();
    if (abs(m1_test - m1) > 0.5) __builtin_trap();
    if (abs(m2_test - m2) > 0.5) __builtin_trap();*/
}

template<typename T>
KERNELSPEC void kernel_two_particle_scattering(
    FIn<T,0> r_phi, FIn<T,1> pa, FIn<T,1> pb, FIn<T,0> t, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,1> p1, FOut<T,1> p2, FOut<T,0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) p_tot[i] = pa[i] + pb[i];
    //auto load_pa = load_mom<T>(pa);
    //auto load_pb = load_mom<T>(pb);
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    //auto pb_com = boost<T>(load_mom<T>(pb), p_tot, -1.);//TODO:remove
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto scatter_out = two_particle_scattering<T>(
        r_phi, pa_com, s_tot, t, m1, m2, ma_2, mb_2
    );
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    //auto m1_test_com = sqrt(lsquare<T>(p1_com));
    auto p1_rot = rotate<T>(p1_com, pa_com);
    //auto m1_test_rot = sqrt(lsquare<T>(p1_rot));
    auto p1_lab = boost<T>(p1_rot, p_tot, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) p2[i] = p_tot[i] - p1_lab[i];
    det = det_tmp;
    /*std::println("LAB pa = {} ma = {}", load_mom<T>(pa), sqrt(lsquare<T>(load_mom<T>(pa))));
    std::println("LAB pb = {} mb = {}", load_mom<T>(pb), sqrt(lsquare<T>(load_mom<T>(pb))));
    std::println("LAB p1 = {} m1 = {}", load_mom<T>(p1), sqrt(lsquare<T>(load_mom<T>(p1))));
    std::println("LAB p2 = {} m2 = {}", load_mom<T>(p2), sqrt(lsquare<T>(load_mom<T>(p2))));
    auto s_tot_test = sqrt(s_tot);
    auto m1_test = sqrt(lsquare<T>(p1_lab));
    auto m2_test = sqrt(lsquare<T>(load_mom<T>(p2)));
    if (s_tot_test < m1 + m2) __builtin_trap();
    if (abs(m1_test - m1) > 0.5) __builtin_trap();
    if (abs(m2_test - m2) > 0.5) __builtin_trap();*/
}

template<typename T>
KERNELSPEC void kernel_t_inv_min_max(
    FIn<T,1> pa, FIn<T,1> pb, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,0> t_min, FOut<T,0> t_max
) {
    // this function is based on the yminmax subroutine in MG5 (genps.f)
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) p_tot[i] = pa[i] + pb[i];

    auto s = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto ysqr = kaellen<T>(s, ma_2, mb_2) * kaellen<T>(s, m1_2, m2_2);
    auto yr = where(ysqr > EPS, sqrt(ysqr), EPS);
    auto m_sum = ma_2 + m1_2;
    auto prod = (s + ma_2 - mb_2) * (s + m1_2 - m2_2);
    auto s_eps = s + EPS;
    auto y1 = m_sum - 0.5 * (prod - yr) / s_eps;
    auto y2 = m_sum - 0.5 * (prod + yr) / s_eps;
    auto t_min_tmp = - where(y1 < y2, y2, y1);
    auto t_max_tmp = - where(y1 < y2, y1, y2);
    t_min = where(t_min_tmp > 0., t_min_tmp, 0.);
    t_max = where(t_max_tmp > t_min, t_max_tmp, t_min + EPS);
    //std::println("args {} {} {} {} {}", s, ma_2, mb_2, static_cast<double>(m1), static_cast<double>(m2));
    //std::println("LIMIT {} {} {} {}", y1, y2, static_cast<double>(t_min), static_cast<double>(t_max));
}

}
}
