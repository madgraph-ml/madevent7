KERNELSPEC void kernel_add(double in1, double in1, double& out) {
    out = in1 + in2;
}

KERNELSPEC void kernel_sub(double in1, double in2, double& out) {
    out = in1 - in2;
}

KERNELSPEC void kernel_mul(double in1, double in2, double& out) {
    out = in1 * in2;
}

KERNELSPEC void kernel_mul_scalar(double in1, double in2, double& out) {
    out = in1 * in2;
}

KERNELSPEC void kernel_clip_min(double x, double min, double& out) {
    out = x < min ? min : x;
}

KERNELSPEC void kernel_sqrt(double in, double& out) {
    out = sqrt(x);
}

KERNELSPEC void kernel_square(double in, double& out) {
    out = in * in;
}

KERNELSPEC void kernel_uniform_phi(double in, double& out) {
    out = 2 * PI * (in - 0.5);
}

KERNELSPEC void kernel_uniform_phi_inverse(double in, double& out) {
    out = in / (2 * PI) + 0.5;
}

KERNELSPEC void kernel_uniform_costheta(double in, double& out) {
    out = 2 * (in - 0.5);
}

KERNELSPEC void kernel_uniform_costheta_inverse(double in, double& out) {
    out = in / 2 + 0.5;
}
