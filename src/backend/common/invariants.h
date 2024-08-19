KERNELSPEC void kernel_uniform_invariant(
    DoubleInput r, DoubleInput s_min, DoubleInput s_max, DoubleOutput s, DoubleOutput gs
) {
    gs = s_max - s_min;
    s = s_min + gs * r;
}

KERNELSPEC void kernel_uniform_invariant_inverse(
    DoubleInput s, DoubleInput s_min, DoubleInput s_max, DoubleOutput r, DoubleOutput gs
) {
    gs = 1 / (s_max - s_min);
    r = (s - s_min) * gs;
}

KERNELSPEC void kernel_breit_wigner_invariant(
    DoubleInput r, DoubleInput mass, DoubleInput width, DoubleInput s_min, DoubleInput s_max, DoubleOutput s, DoubleOutput gs
) {
    auto m2 = mass * mass;
    auto gm = mass * width;
    auto y1 = atan((s_min - m2) / gm);
    auto y2 = atan((s_max - m2) / gm);
    auto dy21 = y2 - y1;
    auto _s = gm * tan(y1 + dy21 * r) + m2;
    auto s_sub_m2 = _s - m2;

    s = _s;
    gs = dy21 * (s_sub_m2 * s_sub_m2 + gm * gm) / gm;
}

KERNELSPEC void kernel_breit_wigner_invariant_inverse(
    DoubleInput s, DoubleInput mass, DoubleInput width, DoubleInput s_min, DoubleInput s_max, DoubleOutput r, DoubleOutput gs
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
    DoubleInput r, DoubleInput mass, DoubleInput s_min, DoubleInput s_max, DoubleOutput s, DoubleOutput gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;

    s = pow(q_max, r) * pow(q_min, 1 - r) + m2;
    gs = (s - m2) * log(q_max / q_min);
}

KERNELSPEC void kernel_stable_invariant_inverse(
    DoubleInput s, DoubleInput mass, DoubleInput s_min, DoubleInput s_max, DoubleOutput r, DoubleOutput gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto q = s - m2;
    auto log_q_max_min = log(q_max / q_min);

    r = log(q / q_min) / log_q_max_min;
    gs = 1 / (q * log_q_max_min);
}

KERNELSPEC void kernel_stable_invariant_nu(
    DoubleInput r, DoubleInput mass, DoubleInput nu, DoubleInput s_min, DoubleInput s_max, DoubleOutput s, DoubleOutput gs
) {
    auto m2 = mass == 0. && s_min == 0 ? -1e-8 : mass * mass;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto power = 1.0 - nu;
    auto qmaxpow = pow(q_max, power);
    auto qminpow = pow(q_min, power);
    auto _s = pow(r * qmaxpow + (1 - r) * qminpow, 1 / power) + m2;

    s = _s;
    gs = (qmaxpow - qminpow) * pow(_s - m2, nu) / power;
}

KERNELSPEC void kernel_stable_invariant_nu_inverse(
    DoubleInput s, DoubleInput mass, DoubleInput nu, DoubleInput s_min, DoubleInput s_max, DoubleOutput r, DoubleOutput gs
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
