#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

template<typename T>
KERNELSPEC void kernel_copy(FIn<T,0> in, FOut<T,0> out) {
    out = in;
}

template<typename T>
KERNELSPEC void kernel_zero(FIn<T,0> in, FOut<T,0> out) {
    out = 0.;
}

template<typename T>
KERNELSPEC void kernel_add_inplace(FIn<T,0> in, FOut<T,0> out) {
    out += in;
}

template<typename T>
KERNELSPEC void kernel_add(FIn<T,0> in1, FIn<T,0> in2, FOut<T,0> out) {
    out.size();
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
KERNELSPEC void kernel_reduce_product(FIn<T,1> in, FOut<T,0> out) {
    FVal<T> product(1.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        product = product * in[i];
    }
    out = product;
}

template<typename T>
KERNELSPEC void backward_kernel_reduce_product(
    FIn<T,1> in, FIn<T,0> out_grad, FOut<T,1> in_grad
) {
    FVal<T> product(1.);
    IVal<T> zero_count(0);
    for (std::size_t i = 0; i < in.size(); ++i) {
        FVal<T> val = in[i];
        auto zero_val = val == 0.;
        product = product * where(zero_val & (zero_count == 0), 1., val);
    }
    auto zero_product = where(zero_count == 0, product, 0.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        FVal<T> val = in[i];
        auto zero_val = val == 0.;
        in_grad[i] += out_grad * where(val == 0., product, zero_product / val);
    }
}

template<typename T>
KERNELSPEC void kernel_sqrt(FIn<T,0> in, FOut<T,0> out) {
    out = sqrt(in);
}

template<typename T>
KERNELSPEC void kernel_square(FIn<T,0> in, FOut<T,0> out) {
    out = in * in;
}

}
}
