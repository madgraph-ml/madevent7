// Helper functions

double _kaellen(double x, double y, double z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

double _costheta_to_invt(
    double s, double p1_2, double p2_2, double m1, double m2, double cos_theta
) {
    // Mandelstam invariant t=(p1-k1)^2 formula (C.21) in https://arxiv.org/pdf/hep-ph/0008033.pdf
    // p=p1+p2 is at rest;
    // p1, p2 are opposite along z-axis
    // k1, k4 are opposite along the direction defined by theta
    // theta is the angle in the COM frame between p1 & k1

    auto m1_2 = m1*m1;
    auto m2_2 = m2*m2;
    auto num1 = (s + m1_2 - m2_2) * (s + p1_2 - p2_2);
    auto num2 = sqrt(_kaellen(s, m1_2, m2_2)) * sqrt(_kaellen(s, p1_2, p2_2)) * cos_theta;
    auto num = num1 - num2;
    auto two_s = 2 * s;
    auto t = m1_2 + p1_2 - num / (two_s >= EPS ? two_s : EPS);
    return t <= -EPS ? t : -EPS;
}


double _invt_to_costheta(
    double s, double p1_2, double p2_2, double m1, double m2, double t
) {
    // https://arxiv.org/pdf/hep-ph/0008033.pdf Eq.(C.21)
    // invert t=(p1-k1)^2 to cos_theta = ...

    auto m1_2 = m1*m1;
    auto m2_2 = m2*m2;
    auto num1 = (t - m1_2 - p1_2) * 2 * s;
    auto num2 = (s + m1_2 - m2_2) * (s + p1_2 - p2_2);
    auto num = num1 + num2;
    auto denom = sqrt(_kaellen(s, m1_2, m2_2)) * sqrt(_kaellen(s, p1_2, p2_2));
    auto cos_theta = num / (denom >= EPS ? denom : EPS);
    return cos_theta < -1.0 ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);
}

// Kernels

KERNELSPEC void kernel_decay_momentum(
    DoubleInput s, DoubleInput sqrt_s, DoubleInput m1, DoubleInput m2,
    DoubleOutput p, DoubleOutput gs
) {
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto sqrt_kaellen = sqrt(_kaellen(s, m1_2, m2_2));
    p[0] = (s + m1_2 - m2_2) / (2 * sqrt_s);
    p[1] = 0;
    p[2] = 0;
    p[3] = sqrt_kaellen / (2 * sqrt_s);
    gs = PI * sqrt_kaellen / (2 * s);
}

KERNELSPEC void kernel_invt_min_max(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2,
    DoubleOutput t_min, DoubleOutput t_max
) {
    t_min = - _costheta_to_invt(s, s_in1, s_in2, m1, m2, 1.0);
    t_max = - _costheta_to_invt(s, s_in1, s_in2, m1, m2, -1.0);
}

KERNELSPEC void kernel_invt_to_costheta(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2,
    DoubleInput t, DoubleOutput cos_theta
) {
    cos_theta = _invt_to_costheta(s, s_in1, s_in2, m1, m2, -t);
}

KERNELSPEC void kernel_costheta_to_invt(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2,
    DoubleInput cos_theta, DoubleOutput t
) {
    t = - _costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_theta);
}

KERNELSPEC void kernel_two_particle_density_inverse(
    DoubleInput s, DoubleInput m1, DoubleInput m2, DoubleOutput gs
) {
    gs = (2 * s) / (PI * sqrt(_kaellen(s, m1 * m1, m2 * m2)));
}

KERNELSPEC void kernel_tinv_two_particle_density(
    DoubleInput det_t, DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleOutput det
) {
    det = det_t * PI / (2 * sqrt(_kaellen(s, s_in1, s_in2)));
}

KERNELSPEC void kernel_tinv_two_particle_density_inverse(
    DoubleInput det_t, DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleOutput det
) {
    det = det_t * 2 * sqrt(_kaellen(s, s_in1, s_in2)) / PI;
}
