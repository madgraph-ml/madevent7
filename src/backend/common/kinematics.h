constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template<typename T>
KERNELSPEC FVal<T> _lsquare(FIn<T,1> p) {
    auto p2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
    return where((p2 > EPS) | (p2 < -EPS), p2, 0.);
}

template<typename T>
KERNELSPEC void _boost(FIn<T,1> k, FIn<T,1> p_boost, FVal<T> sign, FOut<T,1> p_out) {
    // Perform the boost
    // This is in fact a numerical more stable implementation than often used
    auto p2_boost = _lsquare<T>(p_boost);
    auto rsq = sqrt(where(p2_boost < EPS2, EPS2, p2_boost));
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    p_out[0] = e;
    p_out[1] = k[1] + c1 * p_boost[1];
    p_out[2] = k[2] + c1 * p_boost[2];
    p_out[3] = k[3] + c1 * p_boost[3];
}

template<typename T>
KERNELSPEC void _boost_beam(FIn<T,2> q, FVal<T> rapidity, FVal<T> sign, FOut<T,2> p_out) {
    auto cosh_rap = cosh(rapidity);
    auto sinh_rap = sinh(rapidity);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto q_i = q[i];
        auto p_out_i = p_out[i];
        p_out_i[0] = q_i[0] * cosh_rap + sign * q_i[3] * sinh_rap;
        p_out_i[1] = q_i[1];
        p_out_i[2] = q_i[2];
        p_out_i[3] = q_i[3] * cosh_rap + sign * q_i[0] * sinh_rap;
    }
}


// Kernels

template<typename T>
KERNELSPEC void kernel_rotate_zy(
    FIn<T,1> p, FIn<T,0> phi, FIn<T,0> cos_theta, FOut<T,1> q
) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    auto cos_phi = cos(phi);
    auto sin_phi = sin(phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[3] * sin_theta * cos_phi - p[2] * sin_phi;
    q[2] = p[1] * cos_theta * sin_phi + p[3] * sin_theta * sin_phi + p[2] * cos_phi;
    q[3] = p[3] * cos_theta - p[1] * sin_theta;
}

template<typename T>
KERNELSPEC void kernel_rotate_zy_inverse(
    FIn<T,1> p, FIn<T,0> phi, FIn<T,0> cos_theta, FOut<T,1> q
) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    auto cos_phi = cos(phi);
    auto sin_phi = sin(phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[2] * cos_theta * sin_phi - p[3] * sin_theta;
    q[2] = p[2] * cos_phi - p[1] * sin_phi;
    q[3] = p[3] * cos_theta + p[1] * sin_theta * cos_phi + p[2] * sin_theta * sin_phi;
}

template<typename T>
KERNELSPEC void kernel_boost(FIn<T,1> p1, FIn<T,1> p2, FOut<T,1> p_out) {
    _boost<T>(p1, p2, 1.0, p_out);
}

template<typename T>
KERNELSPEC void kernel_boost_inverse(FIn<T,1> p1, FIn<T,1> p2, FOut<T,1> p_out) {
    _boost<T>(p1, p2, -1.0, p_out);
}

template<typename T>
KERNELSPEC void kernel_boost_beam(FIn<T,2> p1, FIn<T,0> rap, FOut<T,2> p_out) {
    _boost_beam<T>(p1, rap, 1.0, p_out);
}

template<typename T>
KERNELSPEC void kernel_boost_beam_inverse(FIn<T,2> p1, FIn<T,0> rap, FOut<T,2> p_out) {
    _boost_beam<T>(p1, rap, -1.0, p_out);
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
    auto p_cms = e_cm / 2;
    p1[0] = p_cms;
    p1[1] = 0;
    p1[2] = 0;
    p1[3] = p_cms;
    p2[0] = p_cms;
    p2[1] = 0;
    p2[2] = 0;
    p2[3] = -p_cms;
}

template<typename T>
KERNELSPEC void kernel_com_angles(FIn<T,1> p, FOut<T,0> phi, FOut<T,0> cos_theta) {
    phi = atan2(p[2], p[1]);
    auto p_mag = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
    cos_theta = p[3] / where(p_mag > EPS, p_mag, EPS);
}

template<typename T>
KERNELSPEC void kernel_s(FIn<T,1> p, FOut<T,0> s) {
    s = _lsquare<T>(p);
}

template<typename T>
KERNELSPEC void kernel_sqrt_s(FIn<T,1> p, FOut<T,0> sqrt_s) {
    auto s = _lsquare<T>(p);
    sqrt_s = sqrt(where(s > 0., s, 0.));
}

template<typename T>
KERNELSPEC void kernel_s_and_sqrt_s(FIn<T,1> p, FOut<T,0> s, FOut<T,0> sqrt_s) {
    auto p2 = _lsquare<T>(p);
    s = p2;
    sqrt_s = sqrt(where(p2 > 0., p2, 0.));
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
KERNELSPEC void kernel_rapidity(FIn<T,0> x1, FIn<T,0> x2, FOut<T,0> rap) {
    rap = 0.5 * log(x1 / x2);
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
