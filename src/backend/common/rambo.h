// constants and helper functions

constexpr double a_fit_vals[] {
    2.0, 2.7120013510837317, 3.0845206333631006, 3.313073324958275, 3.4675119658060622,
    3.57883262988432, 3.662872056324554, 3.7285606321733686, 3.781315678863016, 3.82461375901416
};

void _map_fourvector_rambo_diet(double q0, double cos_theta, double phi, FViewOut<1> q) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    q[0] = q0;
    q[1] = q0 * sin_theta * cos(phi);
    q[2] = q0 * sin_theta * sin(phi);
    q[3] = q0 * cos_theta;
}

double _two_body_decay_factor_massless(double cum_m_prev, double cum_m) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    return 1.0 / (8 * cum_m_prev_square) * (cum_m_prev_square - cum_m * cum_m);
}

double _two_body_decay_factor(double cum_m_prev, double cum_m, double m_prev) {
    auto cum_m_prev_square = cum_m_prev * cum_m_prev;
    auto mass_sum = cum_m + m_prev;
    auto mass_diff = cum_m - m_prev;
    return 1.0 / (8 * cum_m_prev_square) * sqrt(
        (cum_m_prev_square - mass_sum * mass_sum) * (cum_m_prev_square - mass_diff * mass_diff)
    );
}

// Kernels

KERNELSPEC void kernel_fast_rambo_r_to_u(FViewIn<1> r, FViewOut<1> u, FViewOut<0> det) {
    std::size_t n_particles = r.size() + 2;
    double jac = 1;
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
        jac *= (1 - u_i * u_i) / jac_x;
    }
    det = jac;
}

KERNELSPEC void kernel_rambo_four_vectors_massless(
    FViewIn<1> u, FViewIn<0> e_cm, FViewIn<1> cos_theta, FViewIn<1> phi,
    FViewOut<2> ps, FViewOut<2> qs
) {
    double cum_u = 1.;
    double cum_m_prev = e_cm;
    for (std::size_t i = 0; i < ps.size(); ++i) {
        double cum_m;
        if (i == ps.size() - 1) {
            cum_m = 0;
        } else {
            cum_u *= u[i];
            cum_m = e_cm * cum_u;
        }

        auto e_massless = 4 * cum_m_prev * _two_body_decay_factor_massless(cum_m_prev, cum_m);
        auto q_i = qs[i], p_i = ps[i];
        _map_fourvector_rambo_diet(e_massless, cos_theta[i], phi[i], p_i);
        q_i[0] = sqrt(e_massless * e_massless + cum_m * cum_m);
        q_i[1] = -p_i[1];
        q_i[2] = -p_i[2];
        q_i[3] = -p_i[3];

        cum_m_prev = cum_m;
    }
}

KERNELSPEC void kernel_rambo_four_vectors_massive(
    FViewIn<1> u, FViewIn<0> e_cm, FViewIn<1> cos_theta, FViewIn<1> phi, FViewIn<1> masses,
    FViewOut<2> ps, FViewOut<2> qs, FViewOut<0> e_cm_massless, FViewOut<0> det
) {
    double total_mass = 0;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        total_mass += masses[i];
    }
    e_cm_massless = e_cm - total_mass;

    double cum_u = 1.;
    double cum_m_prev = e_cm;
    double cum_k_prev = e_cm_massless;
    double cum_det = 1;
    for (std::size_t i = 0; i < ps.size(); ++i) {
        double cum_m, cum_k;
        auto mass = masses[i];
        if (i == ps.size() - 1) {
            cum_k = 0;
            cum_m = masses[i+1];
        } else {
            cum_u *= u[i];
            cum_k = e_cm_massless * cum_u;
            total_mass -= mass;
            if (total_mass < 0) total_mass = 0;
            cum_m = cum_k + total_mass;
        }

        auto rho_k = _two_body_decay_factor_massless(cum_k_prev, cum_k);
        auto rho_m = _two_body_decay_factor(cum_m_prev, cum_m, mass);
        cum_det *= rho_m / rho_k;
        if (i < ps.size() - 1) {
            cum_det *= cum_m / cum_k;
        }
        auto e_massless = 4 * cum_m_prev * rho_m;
        auto q_i = qs[i], p_i = ps[i];
        _map_fourvector_rambo_diet(e_massless, cos_theta[i], phi[i], p_i);
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
