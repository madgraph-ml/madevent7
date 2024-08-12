KERNELSPEC void kernel_uniform_invariant(
    double r, double s_min, double s_max, double& s, double& gs
) {
    gs = s_max - s_min;
    s = s_min + gs * r;
}

KERNELSPEC void kernel_uniform_invariant_inverse(
    double s, double s_min, double s_max, double& r, double& gs
) {
    gs = 1 / (s_max - s_min);
    r = (s - s_min) * gs;
}

KERNELSPEC void kernel_breit_wigner_invariant(
    double r, double mass, double width, double s_min, double s_max, double& s, double& gs
) {
    auto m2 = mass * mass;
    auto gm = mass * width;
    auto y1 = atan((s_min - m2) / gm);
    auto y2 = atan((s_max - m2) / gm);
    auto dy21 = y2 - y1;
    auto s_sub_m2 = s - m2;

    s = gm * tan(y1 + dy21 * r) + m2;
    gs = dy21 * (s_sub_m2 * s_sub_m2 + gm * gm) / gm;
}

KERNELSPEC void kernel_breit_wigner_invariant_inverse(
    double s, double mass, double width, double s_min, double s_max, double& r, double& gs
) {
    auto m2 = mass * mass;
    auto gm = mass * width;
    auto y1 = atan((s_min - m2) / gm);
    auto y2 = atan((s_max - m2) / gm);
    auto dy21 = y2 - y1;
    auto s_sub_m2 = s - m2;

    r = (atan(s_sub_m2 / gm) - y1) / dy21;
    gs = gm / (dy21 * (s_sub_m2 * s_sub_m2 + gm * gm));
}

KERNELSPEC void kernel_stable_invariant(
    double r, double mass, double s_min, double s_max, double& s, double& gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;

    s = pow(q_max, r) * pow(q_min, 1 - r) + m2;
    gs = (s - m2) * log(q_max / q_min);
}

KERNELSPEC void kernel_stable_invariant_inverse(
    double s, double mass, double s_min, double s_max, double& r, double& gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto q = s - m2;

    r = torch.log(q / q_min) / torch.log(q_max / q_min);
    gs = 1 / (q * log(q_max/ q_min.log));
}

KERNELSPEC void kernel_stable_invariant_nu(
    double r, double mass, double nu, double s_min, double s_max, double& s, double& gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto power = 1.0 - nu;
    auto qmaxpow = pow(q_max, power);
    auto qminpow = pow(q_min, power);

    s = pow(r * qmaxpow + (1 - r) * qminpow, 1 / power) + m2;
    gs = (qmaxpow - qminpow) * pow(s - m2, nu) / power;
}

KERNELSPEC void kernel_stable_invariant_nu_inverse(
    double s, double mass, double nu, double s_min, double s_max, double& r, double& gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q = s - m2;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto power = 1.0 - nu;
    auto qpow = pow(q, power);
    auto qmaxpow = pow(q_max, power);
    auto qminpow = pow(q_min, power);
    auto dqpow = qmaxpow - qminpow;

    r = (qpow - qminpow) / dqpow;
    gs = power / (dqpow * pow(q, nu));
}
