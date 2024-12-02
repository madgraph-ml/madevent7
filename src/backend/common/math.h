KERNELSPEC void kernel_add(FViewIn<1> in1, FViewIn<1> in2, FViewOut<1> out) {
    for (std::size_t i = 0; i < in1.size(); ++i) {
        out[i] = in1[i] + in2[i];
    }
}

KERNELSPEC void kernel_sub(FViewIn<1> in1, FViewIn<1> in2, FViewOut<1> out) {
    for (std::size_t i = 0; i < in1.size(); ++i) {
        out[i] = in1[i] - in2[i];
    }
}

KERNELSPEC void kernel_mul(FViewIn<0> in1, FViewIn<0> in2, FViewOut<0> out) {
    out = in1 * in2;
}

KERNELSPEC void kernel_mul_scalar(FViewIn<1> in1, FViewIn<1> in2, FViewOut<1> out) {
    for (std::size_t i = 0; i < in1.size(); ++i) {
        out[i] = in1[i] * in2[0];
    }
}

KERNELSPEC void kernel_clip_min(FViewIn<0> x, FViewIn<0> min, FViewOut<0> out) {
    out = x < min ? min : x;
}

KERNELSPEC void kernel_sqrt(FViewIn<0> in, FViewOut<0> out) {
    out = sqrt(in);
}

KERNELSPEC void kernel_square(FViewIn<0> in, FViewOut<0> out) {
    out = in * in;
}

KERNELSPEC void kernel_pow(FViewIn<0> in1, FViewIn<0> in2, FViewOut<0> out) {
    out = pow(in1, in2);
}

KERNELSPEC void kernel_uniform_phi(FViewIn<0> in, FViewOut<0> out) {
    out = 2 * PI * (in - 0.5);
}

KERNELSPEC void kernel_uniform_phi_inverse(FViewIn<0> in, FViewOut<0> out) {
    out = in / (2 * PI) + 0.5;
}

KERNELSPEC void kernel_uniform_costheta(FViewIn<0> in, FViewOut<0> out) {
    out = 2 * (in - 0.5);
}

KERNELSPEC void kernel_uniform_costheta_inverse(FViewIn<0> in, FViewOut<0> out) {
    out = in / 2 + 0.5;
}
