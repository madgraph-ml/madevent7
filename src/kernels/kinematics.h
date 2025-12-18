#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template <typename T>
struct FourMom {
    KERNELSPEC FVal<T>& operator[](int i) { return p[i]; }
    KERNELSPEC FVal<T>& operator[](int i) const { return p[i]; }
    FVal<T> p[4];
};

template <typename F, typename S>
struct Pair {
    F first;
    S second;
};

template <typename A, typename B, typename C>
struct Triplet {
    A first;
    B second;
    C third;
};

template <typename A, typename B, typename C, typename D>
struct Quartuplet {
    A first;
    B second;
    C third;
    D fourth;
};

template <typename A, typename B, typename C, typename D, typename E, typename F>
struct Sextuplet {
    A first;
    B second;
    C third;
    D fourth;
    E fifth;
    F sixth;
};

template <
    typename A,
    typename B,
    typename C,
    typename D,
    typename E,
    typename F,
    typename G,
    typename H,
    typename I,
    typename J>
struct Decuplet {
    A first;
    B second;
    C third;
    D fourth;
    E fifth;
    F sixth;
    G seventh;
    H eighth;
    I ninth;
    J tenth;
};

template <typename T>
KERNELSPEC FourMom<T> load_mom(FIn<T, 1> p) {
    return {p[0], p[1], p[2], p[3]};
}

template <typename T>
KERNELSPEC void store_mom(FOut<T, 1> p_to, FourMom<T> p_from) {
    for (int i = 0; i < 4; ++i) {
        p_to[i] = p_from[i];
    }
}

template <typename T>
KERNELSPEC FVal<T> kaellen(FVal<T> x, FVal<T> y, FVal<T> z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

template <typename T>
KERNELSPEC FVal<T> bk_V(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    // Determinant of the 3x3 V-matrix,
    // see Eq.(11) in 10.1103/PhysRev.187.2008.
    // Note: expects the absolute value of t1
    auto a11 = 2.0 * s12;
    auto a12 = ma_2 + s12 - t2;
    auto a13 = s12 + m1_2 - m2_2;
    auto a22 = 2.0 * ma_2;
    auto a23 = ma_2 + m1_2 + t1_abs;
    auto a31 = m0_2 + s12 - m3_2;
    auto a32 = m0_2 + ma_2 - mb_2;

    // Computes the determinant of the 3x3 V-matrix (hard-coded because easier)
    auto det = a12 * a23 * a31 + a12 * a13 * a32 - a11 * a23 * a32 - a13 * a22 * a31;
    return -det / 8.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_gram4(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12,
    FVal<T> s23
) {
    // omputes the 4x4 Gram determinant,
    // see Eq.(B6) in 10.1103/PhysRev.187.2008.
    // Note: expects the absolute value of t1

    // Get upper triangular matrix components which are non-zero
    // as the Gram matrix is symmetric, i.e. (a_{ij} = a_{ji})
    auto a11 = 2.0 * ma_2;
    auto a12 = ma_2 - t1_abs - m1_2;
    auto a13 = ma_2 + t2 - s12;
    auto a14 = ma_2 + mb_2 - m0_2;
    auto a22 = -2.0 * t1_abs;
    auto a23 = t2 - t1_abs - m2_2;
    auto a24 = mb_2 - t1_abs - s23;
    auto a33 = 2.0 * t2;
    auto a34 = t2 + mb_2 - m3_2;
    auto a44 = 2.0 * mb_2;

    // Computes the determinant of the 4x4 Gram matrix (hard-coded because easier)
    auto det = a14 * a23 * a14 * a23 + a13 * a24 * a13 * a24 + a12 * a34 * a12 * a34 -
        a14 * a14 * a22 * a33 - a13 * a13 * a22 * a44 - a12 * a12 * a33 * a44 -
        a23 * a23 * a11 * a44 - a24 * a24 * a11 * a33 - a34 * a34 * a11 * a22 +
        2 * a11 * a23 * a24 * a34 + 2 * a12 * a13 * a23 * a44 +
        2 * a12 * a14 * a24 * a33 + 2 * a13 * a14 * a22 * a34 -
        2 * a12 * a13 * a24 * a34 - 2 * a12 * a14 * a23 * a34 -
        2 * a13 * a14 * a23 * a24 + a11 * a22 * a33 * a44;
    return det / 16.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_sqrt_g3i_g3im1(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    // This is the squaet root of the product of the two 3x3 Gram determinants g3i and
    // g3im1, as in Eq.(11) in 10.1103/PhysRev.187.2008. Note: expects the absolute
    // value of t1
    auto a11 = 2 * s12;
    auto a12 = s12 + ma_2 - t2;
    auto a13 = s12 + m0_2 - m3_2;
    auto b13 = s12 + m1_2 - m2_2;
    auto a22 = 2 * ma_2;
    auto a23 = ma_2 + m0_2 - mb_2;
    auto b23 = ma_2 + m1_2 + t1_abs;
    auto a33 = 2 * m0_2;
    auto b33 = 2 * m1_2;

    // Calculate the two gramm determinants g3i and g3im1
    // (ard-coded because easier)
    auto g3i = a11 * a22 * a33 + 2 * a12 * a23 * a13 - a11 * a23 * a23 -
        a22 * a13 * a13 - a33 * a12 * a12;
    auto g3im1 = a11 * a22 * b33 + 2 * a12 * b23 * b13 - a11 * b23 * b23 -
        a22 * b13 * b13 - b33 * a12 * a12;
    return sqrt(g3i * g3im1) / 8.0;
}

template <typename T>
KERNELSPEC FVal<T> lsquare(FourMom<T> p) {
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
}

template <typename T>
KERNELSPEC FVal<T> esquare(FourMom<T> p) {
    return p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
}

template <typename T>
KERNELSPEC FourMom<T> rotate(FourMom<T> p, FourMom<T> q) {
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq2 = qt2 + q[3] * q[3];
    auto qt = sqrt(max(qt2, EPS2));
    auto qq = sqrt(max(qq2, EPS2));

    // General rotation (valid when qt2 > 0; numerically safe because qt,qq>=eps)
    FourMom<T> r_gen = {
        p[0],
        q[1] * q[3] / (qq * qt) * p[1] - q[2] / qt * p[2] + q[1] / qq * p[3],
        q[2] * q[3] / (qq * qt) * p[1] + q[1] / qt * p[2] + q[2] / qq * p[3],
        -qt / qq * p[1] + q[3] / qq * p[3]
    };

    // Degenerate case qt2 == 0: choose identity for qz>=0 else
    // (px,,py,pz)->(-px,-py,-pz)
    FourMom<T> r_deg_pos = p;
    FourMom<T> r_deg_neg = {p[0], -p[1], p[2], -p[3]};

    auto mask_deg = (qt2 == 0.);
    auto mask_neg = (q[3] < 0.);

    // pick degenerate result depending on sign(qz)
    FourMom<T> r_deg = {
        where(mask_neg, r_deg_neg[0], r_deg_pos[0]),
        where(mask_neg, r_deg_neg[1], r_deg_pos[1]),
        where(mask_neg, r_deg_neg[2], r_deg_pos[2]),
        where(mask_neg, r_deg_neg[3], r_deg_pos[3]),
    };

    // final: if degenerate use r_deg else r_gen
    return {
        where(mask_deg, r_deg[0], r_gen[0]),
        where(mask_deg, r_deg[1], r_gen[1]),
        where(mask_deg, r_deg[2], r_gen[2]),
        where(mask_deg, r_deg[3], r_gen[3]),
    };
}

template <typename T>
KERNELSPEC FourMom<T> rotate_inverse(FourMom<T> p, FourMom<T> q) {
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq2 = qt2 + q[3] * q[3];
    auto qt = sqrt(max(qt2, EPS2));
    auto qq = sqrt(max(qq2, EPS2));

    // General rotation (valid when qt2 > 0; numerically safe because qt,qq>=eps)
    FourMom<T> r_gen = {
        p[0],
        q[1] * q[3] / (qq * qt) * p[1] + q[2] * q[3] / (qq * qt) * p[2] -
            p[3] * qt / qq,
        -q[2] / qt * p[1] + q[1] / qt * p[2],
        q[1] / qq * p[1] + q[2] / qq * p[2] + q[3] / qq * p[3]
    };

    // Degenerate case qt2 == 0: choose identity for qz>=0 else
    // (px,,py,pz)->(-px,-py,-pz)
    FourMom<T> r_deg_pos = p;
    FourMom<T> r_deg_neg = {p[0], -p[1], p[2], -p[3]};

    auto mask_deg = (qt2 == 0.);
    auto mask_neg = (q[3] < 0.);

    // pick degenerate result depending on sign(qz)
    FourMom<T> r_deg = {
        where(mask_neg, r_deg_neg[0], r_deg_pos[0]),
        where(mask_neg, r_deg_neg[1], r_deg_pos[1]),
        where(mask_neg, r_deg_neg[2], r_deg_pos[2]),
        where(mask_neg, r_deg_neg[3], r_deg_pos[3]),
    };

    // final: if degenerate use r_deg else r_gen
    return {
        where(mask_deg, r_deg[0], r_gen[0]),
        where(mask_deg, r_deg[1], r_gen[1]),
        where(mask_deg, r_deg[2], r_gen[2]),
        where(mask_deg, r_deg[3], r_gen[3]),
    };
}

template <typename T>
KERNELSPEC FourMom<T> boost(FourMom<T> k, FourMom<T> p_boost, FVal<T> sign) {
    // Perform the boost
    // This is in fact a numerically more stable implementation than often used
    auto p2_boost = lsquare<T>(p_boost);
    auto rsq = sqrt(max(EPS2, p2_boost));
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    return FourMom<T>{
        e, k[1] + c1 * p_boost[1], k[2] + c1 * p_boost[2], k[3] + c1 * p_boost[3]
    };
}

template <typename T>
KERNELSPEC void
boost_beam(FIn<T, 2> q, FVal<T> x1, FVal<T> x2, FVal<T> sign, FOut<T, 2> p_out) {
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

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> p1com_from_tabs_phi(
    FourMom<T> pa_com,
    FVal<T> s_tot,
    FVal<T> phi,
    FVal<T> t_abs,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> ma_2,
    FVal<T> mb_2
) {
    // this function is based on the gentcms subroutine in MG5 (genps.f)
    // Note: expects t_abs (positive t-ivariant)
    auto m_tot = sqrt(s_tot);

    auto ed = (m1 - m2) * (m1 + m2) / m_tot;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + s_tot;
    auto pp = 0.5 * where(m1 * m2 == 0., m_tot - fabs(ed), sqrt(max(pp2, EPS)));

    auto pa_com_mag =
        sqrt(pa_com[1] * pa_com[1] + pa_com[2] * pa_com[2] + pa_com[3] * pa_com[3]);
    FourMom<T> p1_com;
    auto e1_com = 0.5 * (m_tot + ed);
    p1_com[0] = max(e1_com, 0.);
    p1_com[3] =
        -(m1 * m1 + ma_2 + t_abs - 2. * p1_com[0] * pa_com[0]) / (2.0 * pa_com_mag);
    auto pt2 = pp * pp - p1_com[3] * p1_com[3];
    auto pt = sqrt(max(pt2, 0.));
    p1_com[1] = pt * cos(phi);
    p1_com[2] = pt * sin(phi);

    auto det = PI / (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2)));
    return {p1_com, det};
}

template <typename T>
KERNELSPEC Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> phi_m1_m2_from_p1com(
    FourMom<T> p1_com, FourMom<T> p2, FVal<T> s_tot, FVal<T> ma_2, FVal<T> mb_2
) {
    auto phi = atan2(p1_com[2], p1_com[1]);
    auto det = (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2))) / PI;
    auto m1 = sqrt(max(EPS2, lsquare<T>(p1_com)));
    auto m2 = sqrt(max(EPS2, lsquare<T>(p2)));
    return {phi, m1, m2, det};
}

// Kernels

template <typename T>
KERNELSPEC void
kernel_boost_beam(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, 1.0, p_out);
}

template <typename T>
KERNELSPEC void
kernel_boost_beam_inverse(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, -1.0, p_out);
}

template <typename T>
KERNELSPEC void kernel_com_p_in(FIn<T, 0> e_cm, FOut<T, 1> p1, FOut<T, 1> p2) {
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

template <typename T>
KERNELSPEC void kernel_r_to_x1x2(
    FIn<T, 0> r,
    FIn<T, 0> s_hat,
    FIn<T, 0> s_lab,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> det
) {
    auto tau = s_hat / s_lab;
    x1 = pow(tau, r);
    x2 = pow(tau, (1 - r));
    det = fabs(log(tau)) / s_lab;
}

template <typename T>
KERNELSPEC void kernel_x1x2_to_r(
    FIn<T, 0> x1, FIn<T, 0> x2, FIn<T, 0> s_lab, FOut<T, 0> r, FOut<T, 0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

template <typename T>
KERNELSPEC void kernel_diff_cross_section(
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> pdf1,
    FIn<T, 0> pdf2,
    FIn<T, 0> matrix_element,
    FIn<T, 0> e_cm2,
    FOut<T, 0> result
) {
    result = INV_GEV2_TO_PB * matrix_element * pdf1 * pdf2 /
        (2. * e_cm2 * x1 * x1 * x2 * x2);
}

template <typename T>
KERNELSPEC void kernel_t_inv_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    // this function is based on the yminmax subroutine in MG5 (genps.f)
    // returns the absolute value of the t invariant
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }

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
    auto t_min_tmp = -max(y2, y1);
    auto t_max_tmp = -min(y1, y2);
    t_min = max(t_min_tmp, 0.);
    t_max = where(t_max_tmp > t_min, t_max_tmp, t_min + EPS);
}

template <typename T>
KERNELSPEC void kernel_t_inv_min_max_inverse(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> t_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    // returns the absolute value of the t invariant and its min/max
    FourMom<T> pa1, p_tot;
    for (int i = 0; i < 4; ++i) {
        pa1[i] = pa[i] - p1[i];
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto t_temp = lsquare<T>(pa1);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));

    auto ysqr = kaellen<T>(s, ma_2, mb_2) * kaellen<T>(s, m1_2, m2_2);
    auto yr = where(ysqr > EPS, sqrt(ysqr), EPS);
    auto m_sum = ma_2 + m1_2;
    auto prod = (s + ma_2 - mb_2) * (s + m1_2 - m2_2);
    auto s_eps = s + EPS;
    auto y1 = m_sum - 0.5 * (prod - yr) / s_eps;
    auto y2 = m_sum - 0.5 * (prod + yr) / s_eps;
    auto t_min_tmp = -max(y2, y1);
    auto t_max_tmp = -min(y1, y2);

    t_abs = -t_temp;
    t_min = max(t_min_tmp, 0.);
    t_max = where(t_max_tmp > t_min, t_max_tmp, t_min + EPS);
}

template <typename T>
KERNELSPEC void kernel_s23_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> s_min,
    FOut<T, 0> s_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_tot, p_12, pt2;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
        p_12[i] = pa[i] + pb[i] - p3[i];
        pt2[i] = pb[i] - p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto s12 = lsquare<T>(p_12);
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto t2 = lsquare<T>(pt2);
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = max(kaellen<T>(s12, ma_2, t2), EPS);

    auto sa = m0_2 + m1_2 + 8 * (V + sqrtGG) / (lambda);
    auto sb = m0_2 + m1_2 + 8 * (V - sqrtGG) / (lambda);
    s_min = min(sa, sb);
    s_max = max(sa, sb);
}

template <typename T>
KERNELSPEC void kernel_s23_min_max_inverse(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> s_23,
    FOut<T, 0> s_min,
    FOut<T, 0> s_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_tot, p_12, pt2, p_23;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
        p_12[i] = p1[i] + p2[i];
        pt2[i] = pb[i] - p3[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto s12 = lsquare<T>(p_12);
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto t2 = lsquare<T>(pt2);
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = max(kaellen<T>(s12, ma_2, t2), EPS);

    auto sa = m0_2 + m1_2 + 8 * (V + sqrtGG) / (lambda);
    auto sb = m0_2 + m1_2 + 8 * (V - sqrtGG) / (lambda);
    s_23 = lsquare<T>(p_23);
    s_min = min(sa, sb);
    s_max = max(sa, sb);
}

template <typename T>
KERNELSPEC void kernel_invariants_from_momenta(
    FIn<T, 2> p_ext, FIn<T, 2> factors, FOut<T, 1> invariants
) {
    for (std::size_t i = 0; i < invariants.size(); ++i) {
        auto factors_i = factors[i];
        FourMom<T> p_sum{0., 0., 0., 0.};
        for (std::size_t j = 0; j < p_ext.size(); ++j) {
            auto p_j = p_ext[j];
            auto factor_ij = factors_i[j];
            p_sum[0] = p_sum[0] + factor_ij * p_j[0];
            p_sum[1] = p_sum[1] + factor_ij * p_j[1];
            p_sum[2] = p_sum[2] + factor_ij * p_j[2];
            p_sum[3] = p_sum[3] + factor_ij * p_j[3];
        }
        invariants[i] = lsquare<T>(p_sum);
    }
}

template <typename T>
KERNELSPEC void kernel_sde2_channel_weights(
    FIn<T, 1> invariants,
    FIn<T, 2> masses,
    FIn<T, 2> widths,
    IIn<T, 2> indices,
    FOut<T, 1> channel_weights
) {
    // TODO: MG has a special case here if tprid != 0. check what that is and if we need
    // it...
    FVal<T> channel_weights_norm(0.);
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        auto masses_i = masses[i];
        auto widths_i = widths[i];
        auto indices_i = indices[i];
        FVal<T> prop_product(1.);
        for (std::size_t j = 0; j < indices_i.size(); ++j) {
            auto indices_ij = indices_i[j];
            auto mask = indices_ij == -1;
            auto invar = invariants.gather(where(mask, 0, indices_ij));
            auto mass = masses_i[j];
            auto width = widths_i[j];
            auto tmp = invar - mass * mass;
            auto tmp2 = mass * width;
            prop_product = prop_product * where(mask, 1., tmp * tmp + tmp2 * tmp2);
        }
        auto channel_weight = 1. / prop_product;
        channel_weights[i] = channel_weight;
        channel_weights_norm = channel_weights_norm + channel_weight;
    }
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = channel_weights[i] / channel_weights_norm;
    }
}

template <typename T>
KERNELSPEC void kernel_subchannel_weights(
    FIn<T, 1> invariants,
    FIn<T, 2> masses,
    FIn<T, 2> widths,
    IIn<T, 2> indices,
    IIn<T, 2> on_shell,
    IIn<T, 1> group_sizes,
    FOut<T, 1> channel_weights
) {
    FVal<T> channel_weights_norm(0.);
    std::size_t group_size = single_index(group_sizes[0]);
    std::size_t group_index = 0;
    for (std::size_t i = 0, i_group = 1; i < channel_weights.size(); ++i, ++i_group) {
        FVal<T> channel_weight(1.);
        for (std::size_t j = 0; j < on_shell.size(1); ++j) {
            auto mass = masses[i][j];
            auto width = widths[i][j];
            auto index = indices[i][j];
            auto is_off_shell = on_shell[i][j] == 0;
            auto mask = (index == -1) | is_off_shell;
            auto invar = invariants.gather(where(mask, 0, index));
            auto mass2 = mass * mass;
            auto tmp = invar - mass2;
            auto tmp2 = mass * width;
            channel_weight = channel_weight *
                where(mask,
                      1.,
                      (mass2 * mass2 + tmp2 * tmp2) / (tmp * tmp + tmp2 * tmp2));
        }
        channel_weights[i] = channel_weight;
        channel_weights_norm = channel_weights_norm + channel_weight;
        if (i_group == group_size) {
            for (std::size_t j = 0; j < group_size; ++j) {
                channel_weights[i - j] = channel_weights[i - j] / channel_weights_norm;
            }
            ++group_index;
            group_size = single_index(group_sizes[group_index]);
            i_group = 0;
            channel_weights_norm = 0;
        }
    }
}

template <typename T>
KERNELSPEC void kernel_apply_subchannel_weights(
    FIn<T, 1> channel_weights_in,
    FIn<T, 1> subchannel_weights,
    IIn<T, 1> channel_indices,
    IIn<T, 1> subchannel_indices,
    FOut<T, 1> channel_weights_out
) {
    for (std::size_t i = 0; i < channel_indices.size(); ++i) {
        auto mask = subchannel_indices[i] == -1;
        auto chan_weight = channel_weights_in.gather(channel_indices[i]);
        auto subchan_weight =
            subchannel_weights.gather(where(mask, 0, subchannel_indices[i]));
        channel_weights_out[i] = chan_weight * where(mask, 1., subchan_weight);
    }
}

template <typename T>
KERNELSPEC void
kernel_pt_eta_phi_x(FIn<T, 2> p_ext, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 1> output) {
    output[0] = x1;
    output[1] = x2;
    for (std::size_t i = 2; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto px = p_i[1], py = p_i[2], pz = p_i[3];
        auto pt2 = px * px + py * py + 1e-6;
        output[3 * i - 4] = 0.5 * log(pt2);
        output[3 * i - 3] = atan2(py, px);
        output[3 * i - 2] = atanh(pz / sqrt(pt2 + pz * pz));
    }
}

template <typename T>
KERNELSPEC void
kernel_mirror_momenta(FIn<T, 2> p_ext, IIn<T, 0> mirror, FOut<T, 2> p_out) {
    auto sign = 1. - 2. * FVal<T>(IVal<T>(mirror));
    for (std::size_t i = 0; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto q_i = p_out[i];
        q_i[0] = p_i[0];
        q_i[1] = p_i[1];
        q_i[2] = sign * p_i[2];
        q_i[3] = sign * p_i[3];
    }
}

template <typename T>
KERNELSPEC void
kernel_momenta_to_x1x2(FIn<T, 2> p_ext, FIn<T, 0> e_cm, FOut<T, 0> x1, FOut<T, 0> x2) {
    x1 = 2. * p_ext[0][0] / e_cm;
    x2 = 2. * p_ext[1][0] / e_cm;
}

} // namespace kernels
} // namespace madevent
