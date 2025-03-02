template<typename T>
KERNELSPEC void kernel_copy(FIn<T,0> in, FOut<T,0> out) {
    out = in;
}

template<typename T>
KERNELSPEC void kernel_zero(FIn<T,0> in, FOut<T,0> out) {
    out = 0.;
}

template<typename T>
KERNELSPEC void kernel_add(FIn<T,0> in1, FIn<T,0> in2, FOut<T,0> out) {
    out = in1 + in2;
}

template<typename T>
KERNELSPEC void kernel_sub(FIn<T,0> in1, FIn<T,0> in2, FOut<T,0> out) {
    out = in1 - in2;
}

template<typename T>
KERNELSPEC void kernel_mul(FIn<T,0> in1, FIn<T,0> in2, FOut<T,0> out) {
    out = in1 * in2;
}

template<typename T>
KERNELSPEC void kernel_product(FIn<T,1> in, FOut<T,0> out) {
    FVal<T> product(1.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        product = product * in[i];
    }
    out = product;
}

template<typename T>
KERNELSPEC void kernel_clip_min(FIn<T,0> x, FIn<T,0> min, FOut<T,0> out) {
    out = where(x < min, min, x);
}

template<typename T>
KERNELSPEC void kernel_sqrt(FIn<T,0> in, FOut<T,0> out) {
    out = sqrt(in);
}

template<typename T>
KERNELSPEC void kernel_square(FIn<T,0> in, FOut<T,0> out) {
    out = in * in;
}

template<typename T>
KERNELSPEC void kernel_pow(FIn<T,0> in1, FIn<T,0> in2, FOut<T,0> out) {
    out = pow(in1, in2);
}

template<typename T>
KERNELSPEC void kernel_uniform_phi(FIn<T,0> in, FOut<T,0> out) {
    out = 2 * PI * (in - 0.5);
}

template<typename T>
KERNELSPEC void kernel_uniform_phi_inverse(FIn<T,0> in, FOut<T,0> out) {
    out = in / (2 * PI) + 0.5;
}

template<typename T>
KERNELSPEC void kernel_uniform_costheta(FIn<T,0> in, FOut<T,0> out) {
    out = 2 * (in - 0.5);
}

template<typename T>
KERNELSPEC void kernel_uniform_costheta_inverse(FIn<T,0> in, FOut<T,0> out) {
    out = in / 2 + 0.5;
}
