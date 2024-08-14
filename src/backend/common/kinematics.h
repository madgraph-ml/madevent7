// Helper functions

KERNELSPEC double _lsquare(DoubleInput p) {
    auto p2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
    return p2 > EPS || p2 < -EPS ? p2 : 0.;
}

KERNELSPEC void _boost(DoubleInput k, DoubleInput p_boost, double sign, DoubleOutput p_out) {
    // Perform the boost
    // This is in fact a numerical more stable implementation then often used
    auto p2_boost = _lsquare(p_boost);
    auto rsq = sqrt(p2_boost < EPS2 ? EPS2 : p2_boost);
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    p_out[0] = e;
    p_out[1] = k[1] + c1 * p_boost[1];
    p_out[2] = k[2] + c1 * p_boost[2];
    p_out[3] = k[3] + c1 * p_boost[3];
}

KERNELSPEC void _boost_beam(DoubleInput q, double rapidity, double sign, DoubleOutput p_out) {
    auto cosh_rap = cosh(rapidity);
    auto sinh_rap = sinh(rapidity);
    p_out[0] = q[0] * cosh_rap + sign * q[3] * sinh_rap;
    p_out[1] = q[1];
    p_out[2] = q[2];
    p_out[3] = q[3] * cosh_rap + sign * q[0] * sinh_rap;
}


// Kernels

KERNELSPEC void kernel_rotate_zy(
    DoubleInput p, DoubleInput phi, DoubleInput cos_theta, DoubleOutput q
) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    auto cos_phi = cos(phi);
    auto sin_phi = sin(phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[3] * sin_theta * cos_phi - p[2] * sin_phi;
    q[2] = p[1] * cos_theta * sin_phi + p[3] * sin_theta * sin_phi + p[2] * cos_phi;
    q[3] = p[3] * cos_theta - p[1] * sin_theta;
}

KERNELSPEC void kernel_rotate_zy_inverse(
    DoubleInput p, DoubleInput phi, DoubleInput cos_theta, DoubleOutput q
) {
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);
    auto cos_phi = cos(phi);
    auto sin_phi = sin(phi);

    q[0] = p[0];
    q[1] = p[1] * cos_theta * cos_phi + p[2] * cos_theta * sin_phi - p[3] * sin_theta;
    q[2] = p[2] * cos_phi - p[1] * sin_phi;
    q[3] = p[3] * cos_theta + p[1] * sin_theta * cos_phi + p[2] * sin_theta * sin_phi;
}

KERNELSPEC void kernel_boost(DoubleInput p1, DoubleInput p2, DoubleOutput p_out) {
    _boost(p1, p2, 1.0, p_out);
}

KERNELSPEC void kernel_boost_inverse(DoubleInput p1, DoubleInput p2, DoubleOutput p_out) {
    _boost(p1, p2, -1.0, p_out);
}

KERNELSPEC void kernel_boost_beam(DoubleInput p1, DoubleInput rap, DoubleOutput p_out) {
    _boost_beam(p1, rap, 1.0, p_out);
}

KERNELSPEC void kernel_boost_beam_inverse(DoubleInput p1, DoubleInput rap, DoubleOutput p_out) {
    _boost_beam(p1, rap, -1.0, p_out);
}

KERNELSPEC void kernel_com_momentum(DoubleInput sqrt_s, DoubleOutput p) {
    p[0] = sqrt_s;
    p[1] = 0;
    p[2] = 0;
    p[3] = 0;
}

KERNELSPEC void kernel_com_p_in(DoubleInput e_cm, DoubleOutput p1, DoubleOutput p2) {
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

KERNELSPEC void kernel_com_angles(DoubleInput p, DoubleOutput phi, DoubleOutput cos_theta) {
    phi = atan2(p[2], p[1]);
    cos_theta = p[3] / sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
}

KERNELSPEC void kernel_s(DoubleInput p, DoubleOutput s) {
    s = _lsquare(p);
}

KERNELSPEC void kernel_sqrt_s(DoubleInput p, DoubleOutput sqrt_s) {
    auto s = _lsquare(p);
    sqrt_s = sqrt(s > 0 ? s : 0);
}

KERNELSPEC void kernel_s_and_sqrt_s(DoubleInput p, DoubleOutput s, DoubleOutput sqrt_s) {
    auto p2 = _lsquare(p);
    s = p2;
    sqrt_s = sqrt(p2 > 0 ? p2 : 0);
}

KERNELSPEC void kernel_r_to_x1x2(
    DoubleInput r, DoubleInput s_hat, DoubleInput s_lab, DoubleOutput x1, DoubleOutput x2, DoubleOutput det
) {
    auto tau = s_hat / s_lab;
    x1 = pow(tau, r);
    x2 = pow(tau, (1 - r));
    det = fabs(log(tau)) / s_lab;
}

KERNELSPEC void kernel_x1x2_to_r(
    DoubleInput x1, DoubleInput x2, DoubleInput s_lab, DoubleOutput r, DoubleOutput det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

KERNELSPEC void kernel_rapidity(DoubleInput x1, DoubleInput x2, DoubleOutput rap) {
    rap = 0.5 * log(x1 / x2);
}
