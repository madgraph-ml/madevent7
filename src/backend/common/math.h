KERNELSPEC void kernel_add(DoubleInput in1, DoubleInput in2, DoubleOutput out) {
    out = in1 + in2;
}

KERNELSPEC void kernel_sub(DoubleInput in1, DoubleInput in2, DoubleOutput out) {
    out = in1 - in2;
}

KERNELSPEC void kernel_mul(DoubleInput in1, DoubleInput in2, DoubleOutput out) {
    out = in1 * in2;
}

KERNELSPEC void kernel_mul_scalar(DoubleInput in1, DoubleInput in2, DoubleOutput out) {
    out = in1 * in2;
}

KERNELSPEC void kernel_clip_min(DoubleInput x, DoubleInput min, DoubleOutput out) {
    out = x < min ? min : x;
}

KERNELSPEC void kernel_sqrt(DoubleInput in, DoubleOutput out) {
    out = sqrt(in);
}

KERNELSPEC void kernel_square(DoubleInput in, DoubleOutput out) {
    out = in * in;
}

KERNELSPEC void kernel_uniform_phi(DoubleInput in, DoubleOutput out) {
    out = 2 * PI * (in - 0.5);
}

KERNELSPEC void kernel_uniform_phi_inverse(DoubleInput in, DoubleOutput out) {
    out = in / (2 * PI) + 0.5;
}

KERNELSPEC void kernel_uniform_costheta(DoubleInput in, DoubleOutput out) {
    out = 2 * (in - 0.5);
}

KERNELSPEC void kernel_uniform_costheta_inverse(DoubleInput in, DoubleOutput out) {
    out = in / 2 + 0.5;
}
