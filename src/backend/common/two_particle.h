// Helper functions

template<typename T>
KERNELSPEC FVal<T> _kaellen(FVal<T> x, FVal<T> y, FVal<T> z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

template<typename T>
KERNELSPEC FVal<T> _costheta_to_invt(
    FVal<T> s, FVal<T> p1_2, FVal<T> p2_2, FVal<T> m1, FVal<T> m2, FVal<T> cos_theta
) {
    // Mandelstam invariant t=(p1-k1)^2 formula (C.21) in https://arxiv.org/pdf/hep-ph/0008033.pdf
    // p=p1+p2 is at rest;
    // p1, p2 are opposite along z-axis
    // k1, k4 are opposite along the direction defined by theta
    // theta is the angle in the COM frame between p1 & k1

    auto m1_2 = m1*m1;
    auto m2_2 = m2*m2;
    auto num1 = (s + m1_2 - m2_2) * (s + p1_2 - p2_2);
    auto num2 = sqrt(_kaellen<T>(s, m1_2, m2_2)) * sqrt(_kaellen<T>(s, p1_2, p2_2)) * cos_theta;
    auto num = num1 - num2;
    auto two_s = 2 * s;
    auto t = m1_2 + p1_2 - num / where(two_s >= EPS, two_s, EPS);
    return where(t <= -EPS, t, -EPS);
}


template<typename T>
KERNELSPEC FVal<T> _invt_to_costheta(
    FVal<T> s, FVal<T> p1_2, FVal<T> p2_2, FVal<T> m1, FVal<T> m2, FVal<T> t
) {
    // https://arxiv.org/pdf/hep-ph/0008033.pdf Eq.(C.21)
    // invert t=(p1-k1)^2 to cos_theta = ...

    auto m1_2 = m1*m1;
    auto m2_2 = m2*m2;
    auto num1 = (t - m1_2 - p1_2) * 2 * s;
    auto num2 = (s + m1_2 - m2_2) * (s + p1_2 - p2_2);
    auto num = num1 + num2;
    auto denom = sqrt(_kaellen<T>(s, m1_2, m2_2)) * sqrt(_kaellen<T>(s, p1_2, p2_2));
    auto cos_theta = num / where(denom >= EPS, denom, EPS);
    return where(cos_theta < -1.0, -1.0, where(cos_theta > 1.0, 1.0, cos_theta));
}

// Kernels

template<typename T>
KERNELSPEC void kernel_decay_momentum(
    FIn<T,0> s, FIn<T,0> sqrt_s, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,1> p, FOut<T,0> gs
) {
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto sqrt_kaellen = sqrt(_kaellen<T>(s, m1_2, m2_2));
    p[0] = (s + m1_2 - m2_2) / (2 * sqrt_s);
    p[1] = 0;
    p[2] = 0;
    p[3] = sqrt_kaellen / (2 * sqrt_s);
    gs = PI * sqrt_kaellen / (2 * s);
}

template<typename T>
KERNELSPEC void kernel_invt_min_max(
    FIn<T,0> s, FIn<T,0> s_in1, FIn<T,0> s_in2, FIn<T,0> m1, FIn<T,0> m2,
    FOut<T,0> t_min, FOut<T,0> t_max
) {
    t_min = - _costheta_to_invt<T>(s, s_in1, s_in2, m1, m2, 1.0);
    t_max = - _costheta_to_invt<T>(s, s_in1, s_in2, m1, m2, -1.0);
}

template<typename T>
KERNELSPEC void kernel_invt_to_costheta(
    FIn<T,0> s, FIn<T,0> s_in1, FIn<T,0> s_in2, FIn<T,0> m1, FIn<T,0> m2,
    FIn<T,0> t, FOut<T,0> cos_theta
) {
    cos_theta = _invt_to_costheta<T>(s, s_in1, s_in2, m1, m2, -t);
}

template<typename T>
KERNELSPEC void kernel_costheta_to_invt(
    FIn<T,0> s, FIn<T,0> s_in1, FIn<T,0> s_in2, FIn<T,0> m1, FIn<T,0> m2,
    FIn<T,0> cos_theta, FOut<T,0> t
) {
    t = - _costheta_to_invt<T>(s, s_in1, s_in2, m1, m2, cos_theta);
}

template<typename T>
KERNELSPEC void kernel_two_particle_density_inverse(
    FIn<T,0> s, FIn<T,0> m1, FIn<T,0> m2, FOut<T,0> gs
) {
    gs = (2 * s) / (PI * sqrt(_kaellen<T>(s, m1 * m1, m2 * m2)));
}

template<typename T>
KERNELSPEC void kernel_tinv_two_particle_density(
    FIn<T,0> det_t, FIn<T,0> s, FIn<T,0> s_in1, FIn<T,0> s_in2, FOut<T,0> det
) {
    det = det_t * PI / (2 * sqrt(_kaellen<T>(s, s_in1, s_in2)));
}

template<typename T>
KERNELSPEC void kernel_tinv_two_particle_density_inverse(
    FIn<T,0> det_t, FIn<T,0> s, FIn<T,0> s_in1, FIn<T,0> s_in2, FOut<T,0> det
) {
    det = det_t * 2 * sqrt(_kaellen<T>(s, s_in1, s_in2)) / PI;
}
