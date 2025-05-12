#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

// constants and helper functions

KERNELSPEC double a_fit_vals[] {
    2.0, 2.7120013510837317, 3.0845206333631006, 3.313073324958275, 3.4675119658060622,
    3.57883262988432, 3.662872056324554, 3.7285606321733686, 3.781315678863016, 3.82461375901416
};

template<typename T>
KERNELSPEC void _map_fourvector_rambo_diet(FVal<T> q0, FVal<T> cos_theta, FVal<T> phi, FOut<T,1> q) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    q[0] = q0;
    q[1] = q0 * sin_theta * cos(phi);
    q[2] = q0 * sin_theta * sin(phi);
    q[3] = q0 * cos_theta;
}

template<typename T>
KERNELSPEC FVal<T> _two_body_decay_factor_massless(FVal<T> cum_m_prev, FVal<T> cum_m) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    return 1.0 / (8 * cum_m_prev_square) * (cum_m_prev_square - cum_m * cum_m);
}

template<typename T>
KERNELSPEC FVal<T> _two_body_decay_factor(FVal<T> cum_m_prev, FVal<T> cum_m, FVal<T> m_prev) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    auto mass_sum = cum_m + m_prev;
    auto mass_diff = cum_m - m_prev;
    return 1.0 / (8 * cum_m_prev_square) * sqrt(
        (cum_m_prev_square - mass_sum * mass_sum) * (cum_m_prev_square - mass_diff * mass_diff)
    );
}

// Kernels

template<typename T>
KERNELSPEC void kernel_fast_rambo_r_to_u(FIn<T,1> r, FOut<T,1> u, FOut<T,0> det) {
    std::size_t n_particles = r.size() + 2;
    FVal<T> jac = 1;
    for (std::size_t i = 0, m = n_particles - 2; m > 0; ++i, --m) {
        auto a = a_fit_vals[m - 1];
        auto r_u = r[i];
        auto b = a - r_u * (a - 2);
        auto x = 2 * r_u / (b + sqrt(b * b + 4 * r_u * (1 - b)));
        auto u_i = pow(x, 1.0 / (2 * m));
        auto xr = 1 - x;
        auto jac_denom = (1 + (a - 2) * x * xr);
        auto jac_x = (2 * x * xr + a * xr * xr) / (jac_denom * jac_denom);
        u[i] = u_i;
        jac = jac * (1 - u_i * u_i) / jac_x;
    }
    det = jac;
}

template<typename T>
KERNELSPEC void kernel_rambo_four_vectors_massless(
    FIn<T,1> u, FIn<T,0> e_cm, FIn<T,1> cos_theta, FIn<T,1> phi,
    FOut<T,2> ps, FOut<T,2> qs
) {
    FVal<T> cum_u = 1.;
    FVal<T> cum_m_prev = e_cm;
    for (std::size_t i = 0; i < ps.size(); ++i) {
        FVal<T> cum_m;
        if (i == ps.size() - 1) {
            cum_m = 0;
        } else {
            cum_u = cum_u * u[i];
            cum_m = e_cm * cum_u;
        }

        auto e_massless = 4 * cum_m_prev * _two_body_decay_factor_massless<T>(cum_m_prev, cum_m);
        auto q_i = qs[i], p_i = ps[i];
        _map_fourvector_rambo_diet<T>(e_massless, cos_theta[i], phi[i], p_i);
        q_i[0] = sqrt(e_massless * e_massless + cum_m * cum_m);
        q_i[1] = -p_i[1];
        q_i[2] = -p_i[2];
        q_i[3] = -p_i[3];

        cum_m_prev = cum_m;
    }
}

template<typename T>
KERNELSPEC void kernel_rambo_four_vectors_massive(
    FIn<T,1> u, FIn<T,0> e_cm, FIn<T,1> cos_theta, FIn<T,1> phi, FIn<T,1> masses,
    FOut<T,2> ps, FOut<T,2> qs, FOut<T,0> e_cm_massless, FOut<T,0> det
) {
    FVal<T> total_mass = 0;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total_mass = total_mass + masses[i];
    }
    e_cm_massless = e_cm - total_mass;

    FVal<T> cum_u = 1.;
    FVal<T> cum_m_prev = e_cm;
    FVal<T> cum_k_prev = e_cm_massless;
    FVal<T> cum_det = 1;
    for (std::size_t i = 0; i < ps.size(); ++i) {
        FVal<T> cum_m, cum_k;
        auto mass = masses[i];
        if (i == ps.size() - 1) {
            cum_k = 0;
            cum_m = masses[i+1];
        } else {
            cum_u = cum_u * u[i];
            cum_k = e_cm_massless * cum_u;
            total_mass = max(total_mass - mass, 0.);
            cum_m = cum_k + total_mass;
        }

        auto rho_k = _two_body_decay_factor_massless<T>(cum_k_prev, cum_k);
        auto rho_m = _two_body_decay_factor<T>(cum_m_prev, cum_m, mass);
        cum_det = cum_det * rho_m / rho_k;
        if (i < ps.size() - 1) {
            cum_det = cum_det * cum_m / cum_k;
        }
        auto e_massless = 4 * cum_m_prev * rho_m;
        auto q_i = qs[i], p_i = ps[i];
        _map_fourvector_rambo_diet<T>(e_massless, cos_theta[i], phi[i], p_i);
        p_i[0] = sqrt(e_massless * e_massless + mass * mass);
        q_i[0] = sqrt(e_massless * e_massless + cum_m * cum_m);
        q_i[1] = -p_i[1];
        q_i[2] = -p_i[2];
        q_i[3] = -p_i[3];

        cum_k_prev = cum_k;
        cum_m_prev = cum_m;
    }
    det = cum_det;
}

}
}
