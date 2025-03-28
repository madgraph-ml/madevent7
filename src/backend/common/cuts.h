// Helper functions

template<typename T>
KERNELSPEC FVal<T> _eta(FIn<T,1> p) {
    auto px = p[1], py = p[2], pz = p[3];
    auto p_mag = sqrt(px * px + py * py + pz * pz);
    auto eta = 0.5 * log((p_mag + pz) / (p_mag - pz));
    return where(p_mag < EPS, 99.0, eta);
}

// Kernels

template<typename T>
KERNELSPEC void kernel_cut_unphysical(
    FIn<T,0> w_in, FIn<T,2> p, FIn<T,0> x1, FIn<T,0> x2, FOut<T,0> w_out
) {
    FVal<T> w = where(isnan(w_in), 0., w_in);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto p_i = p[i];
        for (std::size_t j = 0; j < 4; ++j) {
            w = where(isnan(p_i[j]), 0., w);
        }
    }
    w_out = where(
        (x1 < 0.) | (x1 > 1.) | isnan(x1) | (x2 < 0.) | (x2 > 1.) | isnan(x2),
        0.,
        w
    );
}

template<typename T>
KERNELSPEC void kernel_cut_pt(
    FIn<T,2> p, FIn<T,2> min_max, FOut<T,0> w
) {
    FVal<T> ret(1.);
    for (std::size_t i = 0; i < min_max.size(); ++i) {
        auto p_i = p[i + 2];
        auto px = p_i[1], py = p_i[2];
        auto pt2 = px * px + py * py;
        auto min_max_i = min_max[i];
        auto min_i = min_max_i[0], max_i = min_max_i[1];
        ret = where((pt2 < min_i * min_i) | (pt2 > max_i * max_i), 0., ret);
    }
    w = ret;
}

template<typename T>
KERNELSPEC void kernel_cut_eta(
    FIn<T,2> p, FIn<T,2> min_max, FOut<T,0> w
) {
    FVal<T> ret(1.);
    for (std::size_t i = 0; i < min_max.size(); ++i) {
        auto eta = fabs(_eta<T>(p[i + 2]));
        auto min_max_i = min_max[i];
        ret = where((eta < min_max_i[0]) | (eta > min_max_i[1]), 0., ret);
    }
    w = ret;
}

template<typename T>
KERNELSPEC void kernel_cut_dr(
    FIn<T,2> p, IIn<T,2> indices, FIn<T,2> min_max, FOut<T,0> w
) {
    FVal<T> ret(1.);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto indices_i = indices[i];
        auto p1 = p[single_index(indices_i[0]) + 2], p2 = p[single_index(indices_i[1]) + 2];
        auto eta1 = _eta<T>(p1), eta2 = _eta<T>(p2);
        auto phi1 = atan2(p1[2], p1[1]), phi2 = atan2(p2[2], p2[1]);
        auto delta_eta = eta1 - eta2;
        auto delta_phi = phi1 - phi2;
        delta_phi = where(delta_phi >= PI, delta_phi - 2 * PI, delta_phi);
        delta_phi = where(delta_phi < -PI, delta_phi + 2 * PI, delta_phi);
        auto dr2 = delta_eta * delta_eta + delta_phi * delta_phi;
        auto min_max_i = min_max[i];
        auto min_i = min_max_i[0], max_i = min_max_i[1];
        ret = where((dr2 < min_i * min_i) | (dr2 > max_i * max_i), 0., ret);
    }
    w = ret;
}

template<typename T>
KERNELSPEC void kernel_cut_m_inv(
    FIn<T,2> p, IIn<T,2> indices, FIn<T,2> min_max, FOut<T,0> w
) {
    FVal<T> ret(1.);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        FVal<T> e_tot(0.), px_tot(0.), py_tot(0.), pz_tot(0.);
        auto indices_i = indices[i];
        for (std::size_t j = 0; j < indices_i.size(); ++j) {
            auto p_j = p[single_index(indices_i[j]) + 2];
            e_tot = e_tot + p_j[0];
            px_tot = px_tot + p_j[1];
            py_tot = py_tot + p_j[2];
            pz_tot = pz_tot + p_j[3];
        }
        auto m2_inv = e_tot * e_tot - px_tot * px_tot - py_tot * py_tot - pz_tot * pz_tot;
        auto m_inv = sqrt(where(m2_inv < 0., 0., m2_inv));
        auto min_max_i = min_max[i];
        ret = where((m_inv < min_max_i[0]) | (m_inv > min_max_i[1]), 0., ret);
    }
    w = ret;
}

template<typename T>
KERNELSPEC void kernel_cut_sqrt_s(
    FIn<T,0> sqrt_s, FIn<T,1> min_max, FOut<T,0> w
) {
    w = where((sqrt_s < min_max[0]) | (sqrt_s > min_max[1]), 0., 1.);
}
