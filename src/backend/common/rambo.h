// constants and helper functions

constexpr double[10] a_fit_vals{
    2.0, 2.7120013510837317, 3.0845206333631006, 3.313073324958275, 3.4675119658060622,
    3.57883262988432, 3.662872056324554, 3.7285606321733686, 3.781315678863016, 3.82461375901416
};

void map_fourvector_rambo_diet(double q0, double costheta, double phi, FViewOut<1> q) {
    q[0] = q0;
    q[1] = q0 * sqrt(1 - cos_theta**2) * cos(phi);
    q[2] = q0 * sqrt(1 - cos_theta**2) * sin(phi);
    q[3] = q0 * cos_theta;
}

double _two_body_decay_factor_massless(double cum_m_prev, double cum_m, double m_prev) {
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

KERNELSPEC void kernel_fast_rambo_r_to_u(FViewIn<1> r, FViewOut<1> u, FViewOut<1> det) {
    std::size_t n_particles = r.size();

    double jac = 1;
    for (std::size_t i = 0, m = n_particles - 2; m >= 0; ++i, --m) {
        auto a = a_fit_vals[m];
        auto r_u = r[i];
        auto b = a - r_u * (a - 2);
        auto x = 2 * r_u / (b + sqrt(b * b + 4 * r_u * (1 - b)));
        u[i] = pow(x, (1 / (2 * m)));
        auto xr = 1 - x;
        auto jac_x = (2 * x * xr + a * xr * xr) / (1 + (a - 2) * x * xr) ** 2;
        jac *= (1 - u * u) / jac_x;
    }

    det = jac;
}

KERNELSPEC void kernel_rambo_four_vectors_massless(
    FViewIn<1> u, FViewIn<1> e_cm, FViewIn<1> cos_theta, FViewIn<1> phi,
    FViewOut<2> ps, FViewOut<2> qs
) {
    double cum_u = 1.;
    for (std::size_t i = 0; i < ps.size() - 1; ++i) {
        double cum_e;
        if (i == ps.size() - 2) {
            cum_e = 0;
            cum_u *= u[i];
        } else {
            cum_e = e_cm * cum_u;
        }

        auto e_massless = 4 * M[:, :-1] * _two_body_decay_factor_massless(M[:, :-1], M[:, 1:], 0);
        auto q_i = qs[i], p_i = ps[i];
        _map_fourvector_rambo_diet(e_massless, cos_theta, phi, p_i);
        q_i[0] = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
        q_i[1] = -p_i[1];
        q_i[2] = -p_i[2];
        q_i[3] = -p_i[3];
    }
}

KERNELSPEC void kernel_rambo_four_vectors_massive(
    FViewIn<1> u, FViewIn<1> e_cm, FViewIn<1> cos_theta, FViewIn<1> phi,
    FViewOut<2> ps, FViewOut<2> qs, FViewOut<0> e_cm_massless, FViewOut<0> det
) {
    masses_cum = self.masses.flip(0).cumsum(dim=0).flip(0)
    e_cm_massless = e_cm - masses_cum[0]
    K = torch.zeros((r.shape[0], self.nparticles), device=r.device)
    K[:, 0] = e_cm_massless
    K[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm_massless[:, None]
    M = K + masses_cum
    rho_K = two_body_decay_factor(K[:, :-1], K[:, 1:], 0)
    rho_M = two_body_decay_factor(M[:, :-1], M[:, 1:], self.masses[:-1])
    w_m = torch.prod(rho_M / rho_K, dim=1) * torch.prod(M[:, 1:-1] / K[:, 1:-1], dim=1) // / 8
    q = 4 * M[:, :-1] * rho_M
    ps = map_fourvector_rambo_diet(q, cos_theta, phi)
    ps[:, :, 0] = torch.sqrt(ps[:, :, 0]**2 + self.masses[:-1]**2)

    Q0_i = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
    Qp_i = -ps[:, i, 1:]
    Q_i = torch.concat([Q0_i[:, None], Qp_i], dim=1)
}
