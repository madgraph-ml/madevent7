#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

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
KERNELSPEC void kernel_sample_discrete_inverse(
    IIn<T,0> index, IIn<T,0> option_count, FOut<T,0> r, FOut<T,0> det
) {
    IVal<T> opt_count_i(option_count), index_i(index);
    FVal<T> opt_count_f(opt_count_i), index_f(index_i);
    r = (index_f + 0.5) / opt_count_f;
    det = 1. / opt_count_f;
}

template<typename T>
KERNELSPEC void kernel_sample_discrete_probs(
    FIn<T,0> r, FIn<T,1> probs, IOut<T,0> output, FOut<T,0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    FVal<T> cum_prob(0.), prob_out(0.);
    IVal<T> option(0);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / prob_norm;
        cum_prob = cum_prob + prob;
        auto mask = r < cum_prob;
        option = where(mask, IVal<T>(i), option);
        prob_out = where(mask, prob, prob_out);
    }
    output = option;
    det = 1. / prob_out;
}

template<typename T>
KERNELSPEC void kernel_sample_discrete_probs_inverse(
    IIn<T,0> index, FIn<T,1> probs, FOut<T,0> r, FOut<T,0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    FVal<T> cum_prob(0.), random(0.), prob_out(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / prob_norm;
        cum_prob = cum_prob + prob;
        auto mask = index == i;
        random = where(mask, cum_prob + 0.5 * prob, random);
        prob_out = where(mask, prob, prob_out);
    }
    r = random;
    det = prob_out;
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
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = where(i == index, FVal<T>(1.), 0.);
    }
}

template<typename T>
KERNELSPEC void kernel_collect_channel_weights(
    FIn<T,1> amp2, IIn<T,1> channel_indices, IIn<T,0> channel_count, FOut<T,1> channel_weights
) {
    FVal<T> norm(0.);
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = 0.;
    }
    for (std::size_t i = 0; i < amp2.size(); ++i) {
        std::size_t chan_index = single_index(channel_indices[i]);
        if (chan_index >= channel_weights.size()) continue;
        auto amp2_val = amp2[i];
        norm = norm + amp2_val;
        channel_weights[chan_index] += amp2_val;
    }
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = channel_weights[i] / norm;
    }
}

}
}
