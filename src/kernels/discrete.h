#pragma once

#include "definitions.h"

namespace madevent_kernels {

template<typename T>
KERNELSPEC void kernel_sample_discrete(
    FIn<T,0> r, IIn<T,0> option_count, IOut<T,0> output, FOut<T,0> det
) {
    IVal<T> opt_count_i(option_count);
    FVal<T> opt_count_f(opt_count_i);
    IVal<T> option(r * opt_count_f);
    output = option;
    det = opt_count_f;
}

template<typename T>
KERNELSPEC void kernel_sample_discrete_probs(
    FIn<T,0> r, FIn<T,1> probs, IOut<T,0> output, FOut<T,0> det
) {
    FVal<T> cum_prob(0.);
    IVal<T> option(0);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cum_prob = cum_prob + probs[i];
        option = where(r < cum_prob, IVal<T>(i), option);
    }
    output = option;
    det = 1. / probs.gather(option);
}

template<typename T>
KERNELSPEC void kernel_permute_momenta(
    FIn<T,2> momenta, IIn<T,2> permutations, IIn<T,0> index, FOut<T,2> output
) {
    auto perm = permutations[single_index(index)];
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto input_i = momenta[single_index(perm[i])];
        auto output_i = output[i];
        for (std::size_t j = 0; j < 4; ++j) {
            output_i[j] = input_i[j];
        }
    }
}

template<typename T>
KERNELSPEC void kernel_gather(
    IIn<T,0> index, FIn<T,1> choices, FOut<T,0> output
) {
    output = choices.gather(index);
}

template<typename T>
KERNELSPEC void kernel_gather_int(
    IIn<T,0> index, IIn<T,1> choices, IOut<T,0> output
) {
    output = choices.gather(index);
}

template<typename T>
KERNELSPEC void kernel_one_hot(
    IIn<T,0> index, IIn<T,0> option_count, FOut<T,1> output
) {

}

}
