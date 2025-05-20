#pragma once

#include "definitions.h"
#include "kinematics.h"

namespace madevent {
namespace kernels {

// constants and helper functions

// fitted parameters of the rational quadratic function to approximate
// the flat Rambo On Diet transformation
constexpr KERNELSPEC double a_fit_vals[] {
    2.0, 2.7120013510837317, 3.0845206333631006, 3.313073324958275, 3.4675119658060622,
    3.57883262988432, 3.662872056324554, 3.7285606321733686, 3.781315678863016, 3.82461375901416
};

// weight prefactors computed as (pi/2)^(index+2) / Gamma(index+2)
// with index = n_particles - 3 from 0 to 9
constexpr KERNELSPEC double rambo_weight_factor[] {
    2.4674011002723395, 1.9378922925187385, 1.014678031604192, 0.39846313123083515,
    0.12518088458011775, 0.03277227894723081, 0.00735408219871541,
    0.0014439706630862378, 0.00025202042373060593, 3.958727558733292e-05
};

template<typename T>
KERNELSPEC FourMom<T> map_fourvector_rambo_diet(
    FVal<T> q0, FVal<T> r_cos_theta, FVal<T> r_phi
) {
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    return {
        q0,
        q0 * sin_theta * cos(phi),
        q0 * sin_theta * sin(phi),
        q0 * cos_theta
    };
}

template<typename T>
KERNELSPEC FVal<T> two_body_decay_factor_massless(FVal<T> cum_m_prev, FVal<T> cum_m) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    return 1.0 / (8 * cum_m_prev_square) * (cum_m_prev_square - cum_m * cum_m);
}

template<typename T>
KERNELSPEC FVal<T> two_body_decay_factor(FVal<T> cum_m_prev, FVal<T> cum_m, FVal<T> m_prev) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    auto mass_sum = cum_m + m_prev;
    auto mass_diff = cum_m - m_prev;
    return 1.0 / (8 * cum_m_prev_square) * sqrt(
        (cum_m_prev_square - mass_sum * mass_sum) * (cum_m_prev_square - mass_diff * mass_diff)
    );
}

template<typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> fast_rambo_r_to_u(FVal<T> r, std::size_t index) {
    auto a = a_fit_vals[index]; // index = m-1, m = n_part-2, ... , 1
    auto b = a - r * (a - 2.);
    auto x = 2. * r / (b + sqrt(b * b + 4. * r * (1 - b)));
    auto u = pow(x, 1.0 / (2 * index + 2));
    auto xr = 1. - x;
    auto det_denom = (1. + (a - 2.) * x * xr);
    auto det_x = (2. * x * xr + a * xr * xr) / (det_denom * det_denom);
    auto det = (1. - u * u) / det_x;
    return {u, det};
}

template<typename T>
KERNELSPEC void fast_rambo_massless_body(
    FIn<T,1> r, FIn<T,0> e_cm, FOut<T,2> p_out, FOut<T,0> det, FourMom<T> q
) {
    std::size_t n_particles = p_out.size();
    FVal<T> det_tmp = rambo_weight_factor[n_particles - 3];
    FVal<T> cum_u = 1.;
    FVal<T> cum_m_prev = e_cm;
    for (std::size_t i = 0; i < n_particles - 1; ++i) {
        FVal<T> cum_m;
        if (i == n_particles - 2) {
            cum_m = 0;
        } else {
            auto [u, det_u] = fast_rambo_r_to_u<T>(r[3*i+2], n_particles - 3 - i);
            det_tmp = det_tmp * det_u;
            cum_u = cum_u * u;
            cum_m = e_cm * cum_u;
        }

        auto e_massless = 4 * cum_m_prev * two_body_decay_factor_massless<T>(cum_m_prev, cum_m);
        auto p_i = map_fourvector_rambo_diet<T>(e_massless, r[3*i], r[3*i+1]);
        FourMom<T> q_i {
            sqrt(e_massless * e_massless + cum_m * cum_m), -p_i[1], -p_i[2], -p_i[3]
        };
        store_mom<T>(p_out[i], boost<T>(p_i, q, 1.0));
        q = boost<T>(q_i, q, 1.0);

        cum_m_prev = cum_m;
    }
    store_mom<T>(p_out[n_particles - 1], q);
    det = det_tmp * pow(e_cm, 2 * n_particles - 4);
}

template<typename T>
KERNELSPEC void fast_rambo_massive_body(
    FIn<T,1> r, FIn<T,0> e_cm, FIn<T,1> masses, FOut<T,2> p_out, FOut<T,0> det, FourMom<T> q
) {
    FVal<T> total_mass = 0.;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total_mass = total_mass + masses[i];
    }
    auto e_cm_massless = e_cm - total_mass;

    std::size_t n_particles = p_out.size();
    FVal<T> det_tmp = rambo_weight_factor[n_particles - 3];
    FVal<T> cum_u = 1.;
    FVal<T> cum_m_prev = e_cm;
    FVal<T> cum_k_prev = e_cm_massless;
    for (std::size_t i = 0; i < n_particles - 1; ++i) {
        FVal<T> cum_m, cum_k;
        auto mass = masses[i];
        if (i == n_particles - 2) {
            cum_k = 0;
            cum_m = masses[i+1];
        } else {
            auto [u, det_u] = fast_rambo_r_to_u<T>(r[3*i+2], n_particles - 3 - i);
            det_tmp = det_tmp * det_u;
            cum_u = cum_u * u;
            cum_k = e_cm_massless * cum_u;
            total_mass = max(total_mass - mass, 0.);
            cum_m = cum_k + total_mass;
        }

        auto rho_k = two_body_decay_factor_massless<T>(cum_k_prev, cum_k);
        auto rho_m = two_body_decay_factor<T>(cum_m_prev, cum_m, mass);
        det_tmp = det_tmp * rho_m / rho_k;
        if (i < n_particles - 2) {
            det_tmp = det_tmp * cum_m / cum_k;
        }
        auto e_massless = 4 * cum_m_prev * rho_m;
        auto p_i = map_fourvector_rambo_diet<T>(e_massless, r[3*i], r[3*i+1]);
        p_i[0] = sqrt(e_massless * e_massless + mass * mass);
        FourMom<T> q_i {
            sqrt(e_massless * e_massless + cum_m * cum_m), -p_i[1], -p_i[2], -p_i[3]
        };
        store_mom<T>(p_out[i], boost<T>(p_i, q, 1.0));
        q = boost<T>(q_i, q, 1.0);

        cum_k_prev = cum_k;
        cum_m_prev = cum_m;
    }
    store_mom<T>(p_out[n_particles - 1], q);
    det = det_tmp * pow(e_cm_massless, 2 * n_particles - 4);
}

// Kernels

template<typename T>
KERNELSPEC void kernel_fast_rambo_massless(
    FIn<T,1> r, FIn<T,0> e_cm, FIn<T,1> p0, FOut<T,2> p_out, FOut<T,0> det
) {
    fast_rambo_massless_body<T>(r, e_cm, p_out, det, load_mom<T>(p0));
}

template<typename T>
KERNELSPEC void kernel_fast_rambo_massless_com(
    FIn<T,1> r, FIn<T,0> e_cm, FOut<T,2> p_out, FOut<T,0> det
) {
    FourMom<T> p0 {e_cm, 0., 0., 0.};
    fast_rambo_massless_body<T>(r, e_cm, p_out, det, p0);
}

template<typename T>
KERNELSPEC void kernel_fast_rambo_massive(
    FIn<T,1> r, FIn<T,0> e_cm, FIn<T,1> masses, FIn<T,1> p0, FOut<T,2> p_out, FOut<T,0> det
) {
    fast_rambo_massive_body<T>(r, e_cm, masses, p_out, det, load_mom<T>(p0));
}

template<typename T>
KERNELSPEC void kernel_fast_rambo_massive_com(
    FIn<T,1> r, FIn<T,0> e_cm, FIn<T,1> masses, FOut<T,2> p_out, FOut<T,0> det
) {
    FourMom<T> p0 {e_cm, 0., 0., 0.};
    fast_rambo_massive_body<T>(r, e_cm, masses, p_out, det, p0);
}

}
}
