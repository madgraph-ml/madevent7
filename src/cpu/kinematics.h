#pragma once

#include <cmath>

namespace madevent {
namespace cpu {

const double EPS = 1e-12;
const double EPS2 = 1e-24;


// Helper functions

inline double _lsquare(const Accessor p) {
    auto p2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
    return p2 > -EPS && p2 < EPS ? 0. : p2;
}

inline void _boost(const Accessor k, const Accessor p_boost, double sign, Accessor p_out) {
    // Perform the boost
    // This is in fact a numerical more stable implementation then often used
    auto p2_boost = _lsquare(p_boost);
    auto rsq = std::sqrt(p2_boost < EPS2 ? EPS2 : p2_boost);
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    p_out[0] = e;
    p_out[1] = k[1] + c1 * p_boost[1];
    p_out[2] = k[2] + c1 * p_boost[2];
    p_out[3] = k[3] + c1 * p_boost[3];
}


inline void _boost_beam(const Accessor q, double rapidity, double sign, Accessor p_out) {
    auto cosh_rap = std::cosh(rapidity);
    auto sinh_rap = std::sinh(rapidity);
    p_out[0] = q[0] * cosh_rap + sign * q[3] * sinh_rap;
    p_out[1] = q[1];
    p_out[2] = q[2];
    p_out[3] = q[3] * cosh_rap + sign * q[0] * sinh_rap;
}


// Kernels

inline void rotate_zy(
    const Accessor p, const Accessor phi, const Accessor _cos_theta, Accessor q
) {
    auto cos_theta = *_cos_theta;
    auto sin_theta = std::sqrt(1 - costheta**2);
    auto cos_phi = std::cos(*phi);
    auto sin_phi = std::sin(*phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[3] * sin_theta * cos_phi - p[2] * sin_phi;
    q[2] = p[1] * cos_theta * sin_phi + p[3] * sin_theta * sin_phi + p[2] * cos_phi;
    q[3] = p[3] * cos_theta - p[1] * sin_theta;
}

inline void rotate_zy_inverse(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    auto cos_theta = *_cos_theta;
    auto sin_theta = std::sqrt(1 - costheta**2);
    auto cos_phi = std::cos(*phi);
    auto sin_phi = std::sin(*phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[2] * cos_theta * sin_phi - p[3] * sin_theta;
    q[2] = p[2] * cos_phi - p[1] * sin_phi;
    q[3] = p[3] * cos_theta + p[1] * sin_theta * cos_phi + p[2] * sin_theta * sin_phi;
}

inline void boost(const Accessor p1, const Accessor p2, Accessor p_out) {
    _boost(p1, p2, 1.0, p_out);
}

inline void boost_inverse(const Accessor p1, const Accessor p2, Accessor p_out) {
    _boost(p1, p2, -1.0, p_out);
}

inline void boost_beam(const Accessor p1, const Accessor rap, Accessor p_out) {
    _boost_beam(p1, *rap, 1.0, p_out);
}

inline void boost_beam_inverse(const Accessor p1, const Accessor rap, Accessor p_out) {
    _boost_beam(p1, *rap, -1.0, p_out);
}

inline void com_momentum(const Accessor sqrt_s, Accessor p) {
    p[0] = *sqrt_s;
    p[1] = 0;
    p[2] = 0;
    p[3] = 0;
}

inline void com_p_in(const Accessor e_cm, Accessor p1, Accessor p2) {
    auto p_cms = *e_cm / 2;
    p1[0] = p_cms;
    p1[1] = 0;
    p1[2] = 0;
    p1[3] = p_cms;
    p2[0] = p_cms;
    p2[1] = 0;
    p2[2] = 0;
    p2[3] = -p_cms;
}

inline void com_angles(const Accessor p, Accessor phi, Accessor costheta) {
    *phi = std::atan2(p[2], p[1]);
    *costheta = p[3] / std::sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
}

inline void s(const Accessor p, Accessor _s) {
    *_s = _lsquare(p);
}

inline void sqrt_s(const Accessor p, Accessor _sqrt_s) {
    auto s = _lsquare(p);
    *_sqrt_s = std::sqrt(s > 0 ? s : 0);
}

inline void s_and_sqrt_s(const Accessor p, Accessor s, Accessor sqrt_s) {
    auto _s = _lsquare(p);
    *s = _s;
    *sqrt_s = std::sqrt(_s > 0 ? _s : 0);
}

inline void r_to_x1x2(
    const Accessor _r, const Accessor s_hat, const Accessor _s_lab,
    Accessor x1, Accessor x2, Accessor det
) {
    auto r = *_r, s_lab = *_s_lab;
    auto tau = *s_hat / s_lab;
    *x1 = tau ** r;
    *x2 = tau ** (1 - r);
    *det = std::fabs(std::log(tau)) / s_lab;
}

inline void x1x2_to_r(
    const Accessor _x1, const Accessor _x2, const Accessor s_lab, Accessor r, Accessor det
) {
    auto x1 = *x1, x2 = *x2;
    tau = x1 * x2;
    log_tau = std::log(tau);
    *r = std::log(x1) / log_tau;
    *det = std::fabs(1 / log_tau) * *s_lab;
}

inline void rapidity(const Accessor x1, const Accessor x2, Accessor rap) {
    *rap = 0.5 * std::log(*x1 / *x2);
}

}
}
