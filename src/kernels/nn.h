#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

inline constexpr double MIN_BIN_SIZE = 1e-3;
inline constexpr double MIN_DERIVATIVE = 1e-3;

// Kernels

template<typename T>
KERNELSPEC void kernel_leaky_relu(FIn<T,0> input, FOut<T,0> output) {
    FVal<T> x = input;
    output = where(x < 0, x * 0.01, x);
}

template<typename T>
KERNELSPEC void backward_kernel_leaky_relu(
    FIn<T,0> input, FIn<T,0> output_grad, FOut<T,0> input_grad
) {
    FVal<T> x(input), g(output_grad);
    input_grad = where(x < 0, g * 0.01, g);
}

template<typename T>
KERNELSPEC void kernel_rqs_activation(
    FIn<T,1> input, IIn<T,0> bin_count,
    FOut<T,2> widths, FOut<T,2> heights, FOut<T,2> derivatives
) {
    std::size_t n_bins = single_index(bin_count);
    std::size_t n_cond = 3 * n_bins + 1;
    std::size_t n_dims = input.size() / n_cond;

    for (std::size_t j = 0; j < n_dims; ++j) {
        std::size_t offset = j * n_cond;
        auto w_out = widths[j];
        FVal<T> w_norm(0.);
        for (std::size_t i = 0; i < n_bins; ++i) {
            auto w = exp(input[offset + i]);
            w_norm = w_norm + w;
            w_out[i] = w;
        }
        for (std::size_t i = 0; i < n_bins; ++i) {
            w_out[i] = w_out[i] / w_norm;
        }

        offset += n_bins;
        auto h_out = heights[j];
        FVal<T> h_norm(0.);
        for (std::size_t i = 0; i < n_bins; ++i) {
            auto h = exp(input[offset + i]);
            h_norm = h_norm + h;
            h_out[i] = h;
        }
        for (std::size_t i = 0; i < n_bins; ++i) {
            h_out[i] = h_out[i] / h_norm;
        }

        offset += n_bins;
        auto d_out = derivatives[j];
        for (std::size_t i = 0; i < n_bins + 1; ++i) {
            d_out[i] = input[offset + i];
        }
    }
}

template<typename T>
KERNELSPEC void backward_kernel_rqs_activation(
    FIn<T,2> widths, FIn<T,2> heights,
    FIn<T,2> widths_grad, FIn<T,2> heights_grad, FIn<T,2> derivatives_grad,
    FOut<T,1> input_grad, FOut<T,0> bin_count_grad
) {
    std::size_t n_dims = widths.size();
    std::size_t n_bins = (input_grad.size() / n_dims - 1) / 3;
    std::size_t n_cond = 3 * n_bins + 1;

    for (std::size_t j = 0; j < n_dims; ++j) {
        auto w_j = widths[j];
        auto grad_w_j = widths_grad[j];
        std::size_t offset = j * n_cond;
        FVal<T> w_grad_sum(0.);
        for (std::size_t i = 0; i < n_bins; ++i) {
            w_grad_sum = w_grad_sum + w_j[i] * grad_w_j[i];
        }
        for (std::size_t i = 0; i < n_bins; ++i) {
            input_grad[offset + i] += w_j[i] * (grad_w_j[i] - w_grad_sum);
        }

        auto h_j = heights[j];
        auto grad_h_j = heights_grad[j];
        offset += n_bins;
        FVal<T> h_grad_sum(0.);
        for (std::size_t i = 0; i < n_bins; ++i) {
            h_grad_sum = h_grad_sum + h_j[i] * grad_h_j[i];
        }
        for (std::size_t i = 0; i < n_bins; ++i) {
            input_grad[offset + i] += h_j[i] * (grad_h_j[i] - h_grad_sum);
        }

        offset += n_bins;
        for (std::size_t i = 0; i < n_bins + 1; ++i) {
            input_grad[offset + i] += derivatives_grad[j][i];
        }
    }
}

template<typename T>
KERNELSPEC void kernel_rqs_find_bin(
    FIn<T,0> input, FIn<T,1> in_sizes, FIn<T,1> out_sizes, FIn<T,1> derivatives,
    FOut<T,1> condition
) {
    auto n_bins = in_sizes.size();
    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));
    auto bin_factor = 1. - MIN_BIN_SIZE * n_bins;

    FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
    FVal<T> width(0.), height(0.), cumwidth(0.), cumheight(0.);
    FVal<T> derivative_unorm(0.), derivative_plus_one_unorm(0.);
    for (std::size_t bin = 0; bin < n_bins; ++bin) {
        auto w = MIN_BIN_SIZE + bin_factor * in_sizes[bin];
        auto h = MIN_BIN_SIZE + bin_factor * out_sizes[bin];
        auto d = derivatives[bin];
        auto dp1 = derivatives[bin + 1];

        auto mask = input01 < loop_cumwidth;
        width = where(mask, width, w);
        height = where(mask, height, h);
        derivative_unorm = where(mask, derivative_unorm, d);
        derivative_plus_one_unorm = where(mask, derivative_plus_one_unorm, dp1);
        cumwidth = where(mask, cumwidth, loop_cumwidth);
        cumheight = where(mask, cumheight, loop_cumheight);
        loop_cumwidth = loop_cumwidth + w;
        loop_cumheight = loop_cumheight + h;
    }
    condition[0] = width;
    condition[1] = height;
    condition[2] = cumwidth;
    condition[3] = cumheight;
    condition[4] = derivative_unorm;
    condition[5] = derivative_plus_one_unorm;
}

template<typename T>
KERNELSPEC void backward_kernel_rqs_find_bin(
    FIn<T,0> input, FIn<T,1> in_sizes, FIn<T,1> out_sizes, FIn<T,1> derivatives,
    FIn<T,1> condition_grad, FOut<T,0> input_grad, FOut<T,1> in_sizes_grad,
    FOut<T,1> out_sizes_grad, FOut<T,1> derivatives_grad
) {
    FVal<T> width_grad = condition_grad[0];
    FVal<T> height_grad = condition_grad[1];
    FVal<T> cumwidth_grad = condition_grad[2];
    FVal<T> cumheight_grad = condition_grad[3];
    FVal<T> derivative_unorm_grad = condition_grad[4];
    FVal<T> derivative_plus_one_unorm_grad = condition_grad[5];
    auto n_bins = in_sizes.size();
    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));
    auto bin_factor = 1. - MIN_BIN_SIZE * n_bins;

    FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
    IVal<T> selected_bin(0);
    BVal<T> mask(false);
    for (std::size_t bin = 0; bin < n_bins; ++bin) {
        auto w = MIN_BIN_SIZE + bin_factor * in_sizes[bin];
        selected_bin = where(mask, selected_bin, bin);
        loop_cumwidth = loop_cumwidth + w;
        mask = input01 < loop_cumwidth;
        in_sizes_grad[bin] += where(mask, 0., bin_factor * cumwidth_grad);
        out_sizes_grad[bin] += where(mask, 0., bin_factor * cumheight_grad);
    }
    in_sizes_grad.scatter_add(selected_bin, bin_factor * width_grad);
    out_sizes_grad.scatter_add(selected_bin, bin_factor * height_grad);
    derivatives_grad.scatter_add(selected_bin, derivative_unorm_grad);
    derivatives_grad.scatter_add(selected_bin + 1, derivative_plus_one_unorm_grad);
}

template<typename T>
KERNELSPEC void kernel_rqs_forward(
    FIn<T,0> input, FIn<T,1> condition, FOut<T,0> output, FOut<T,0> det
) {
    FVal<T> width(condition[0]), height(condition[1]);
    FVal<T> cumwidth(condition[2]), cumheight(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = (where(
        derivative_unorm > 20., derivative_unorm, log1p(exp(derivative_unorm))
    ) + MIN_DERIVATIVE) / softplus_scale;
    auto derivative_plus_one = (where(
        derivative_plus_one_unorm > 20.,
        derivative_plus_one_unorm,
        log1p(exp(derivative_plus_one_unorm))
    ) + MIN_DERIVATIVE) / softplus_scale;

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto delta = height / width;
    auto input_diff = input01 - cumwidth;
    auto theta = input_diff / width;
    auto one_minus_theta = 1. - theta;
    auto theta_theta = theta * theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto numerator = height * (delta * theta_theta + derivative * theta_one_minus_theta);
    auto two_delta = 2. * delta;
    auto denominator = delta +
        (derivative + derivative_plus_one - two_delta) * theta_one_minus_theta;
    auto out = cumheight + numerator / denominator;
    output = where(clamp, input, out);

    auto derivative_numerator = delta * delta * (
        derivative_plus_one * theta_theta +
        two_delta * theta_one_minus_theta +
        derivative * one_minus_theta * one_minus_theta
    );
    det = where(clamp, 1., derivative_numerator / (denominator * denominator));
}

template<typename T>
KERNELSPEC void backward_kernel_rqs_forward(
    FIn<T,0> input, FIn<T,1> condition,
    FIn<T,0> output_grad, FIn<T,0> det_grad,
    FOut<T,0> input_grad, FOut<T,1> condition_grad
) {
    FVal<T> width(condition[0]), height(condition[1]);
    FVal<T> cumwidth(condition[2]), cumheight(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto expd = exp(derivative_unorm);
    auto expdpo = exp(derivative_plus_one_unorm);
    auto log1pd = log1p(expd);
    auto log1pdpo = log1p(expdpo);
    auto spd = where(derivative_unorm > 20., derivative_unorm, log1pd);
    auto spdpo = where(
        derivative_plus_one_unorm > 20., derivative_plus_one_unorm, log1pdpo
    );
    auto spd_md = spd + MIN_DERIVATIVE;
    auto spdpo_md = spdpo + MIN_DERIVATIVE;
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = spd_md / softplus_scale;
    auto derivative_plus_one = spdpo_md / softplus_scale;

    auto delta = height / width;
    auto input_diff = input01 - cumwidth;
    auto theta = input_diff / width;
    auto one_minus_theta = 1. - theta;
    auto theta_theta = theta * theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto tmp1 = delta * theta_theta;
    auto tmp2 = derivative * theta_one_minus_theta;
    auto tmp3 = tmp1 + tmp2;
    auto numerator = height * tmp3;
    auto two_delta = 2. * delta;
    auto tmp4 = derivative + derivative_plus_one;
    auto tmp5 = tmp4 - two_delta;
    auto tmp6 = tmp5 * theta_one_minus_theta;
    auto denominator = delta + tmp6;
    auto tmp7 = numerator / denominator;
    auto out = cumheight + tmp7;

    auto tmp8 = delta * delta;
    auto tmp9 = derivative_plus_one * theta_theta;
    auto tmp10 = two_delta * theta_one_minus_theta;
    auto tmp11 = derivative * one_minus_theta;
    auto tmp12 = tmp11 * one_minus_theta;
    auto tmp13 = tmp9 + tmp10;
    auto tmp14 = tmp13 + tmp12;
    auto derivative_numerator = tmp8 * tmp14;
    auto tmp15 = denominator * denominator;
    auto tmp16 = derivative_numerator / tmp15;

    auto grad_tmp16 = det_grad;
    auto grad_tmp15 = - grad_tmp16 * derivative_numerator / (tmp15 * tmp15);
    auto grad_derivative_numerator = grad_tmp16 / tmp15;
    auto grad_denominator = 2. * grad_tmp15 * denominator;
    auto grad_tmp8 = tmp14 * grad_derivative_numerator;
    auto grad_tmp14 = tmp8 * grad_derivative_numerator;
    auto grad_tmp13 = grad_tmp14;
    auto grad_tmp12 = grad_tmp14;
    auto grad_tmp9 = grad_tmp13;
    auto grad_tmp10 = grad_tmp13;
    auto grad_tmp11 = one_minus_theta * grad_tmp12;
    auto grad_one_minus_theta = tmp11 * grad_tmp12;
    auto grad_derivative = one_minus_theta * grad_tmp11;
    grad_one_minus_theta += derivative * grad_tmp11;
    auto grad_two_delta = theta_one_minus_theta * grad_tmp10;
    auto grad_theta_one_minus_theta = two_delta * grad_tmp10;
    auto grad_derivative_plus_one = theta_theta * grad_tmp9;
    auto grad_theta_theta = derivative_plus_one * grad_tmp9;
    auto grad_delta = 2. * delta * grad_tmp8;

    auto grad_cumheight = output_grad;
    auto grad_tmp7 = output_grad;
    auto grad_numerator = grad_tmp7 / denominator;
    grad_denominator += - numerator * grad_tmp7 / (denominator * denominator);
    grad_delta += grad_denominator;
    auto grad_tmp6 = grad_denominator;
    auto grad_tmp5 = theta_one_minus_theta * grad_tmp6;
    grad_theta_one_minus_theta += tmp5 * grad_tmp6;
    auto grad_tmp4 = grad_tmp5;
    grad_two_delta += -grad_tmp5;
    grad_derivative += grad_tmp4;
    grad_derivative_plus_one += grad_tmp4;
    grad_delta += 2. * grad_two_delta;
    auto grad_height = tmp3 * grad_numerator;
    auto grad_tmp3 = height * grad_numerator;
    auto grad_tmp1 = grad_tmp3;
    auto grad_tmp2 = grad_tmp3;
    grad_derivative += theta_one_minus_theta * grad_tmp2;
    grad_theta_one_minus_theta += derivative * grad_tmp2;
    grad_delta += theta_theta * grad_tmp1;
    grad_theta_theta += delta * grad_tmp1;
    auto grad_theta = one_minus_theta * grad_theta_one_minus_theta;
    grad_one_minus_theta += theta * grad_theta_one_minus_theta;
    grad_theta += 2. * theta * grad_theta_theta;
    grad_theta += -grad_one_minus_theta;
    auto grad_input_diff = grad_theta / width;
    auto grad_width = - input_diff * grad_theta / (width * width);
    auto grad_input = grad_input_diff;
    auto grad_cumwidth = -grad_input_diff;

    grad_height += grad_delta / width;
    grad_width += - height * grad_delta / (width * width);

    auto grad_spdpo_md = grad_derivative_plus_one / softplus_scale;
    auto grad_spd_md = grad_derivative / softplus_scale;
    auto grad_spdpo = grad_spdpo_md;
    auto grad_spd = grad_spd_md;

    auto grad_log1pdpo = grad_spdpo;
    auto grad_expdpo = grad_log1pdpo / (1. + expdpo);
    auto grad_derivative_plus_one_unorm = where(
        derivative_plus_one_unorm > 20., grad_spdpo, expdpo * grad_expdpo
    );

    auto grad_log1pd = grad_spd;
    auto grad_expd = grad_log1pd / (1. + expd);
    auto grad_derivative_unorm = where(derivative_unorm > 20., grad_spd, expd * grad_expd);

    input_grad = where(clamp, output_grad, grad_input);
    condition_grad[0] = where(clamp, 0., grad_width);
    condition_grad[1] = where(clamp, 0., grad_height);
    condition_grad[2] = where(clamp, 0., grad_cumwidth);
    condition_grad[3] = where(clamp, 0., grad_cumheight);
    condition_grad[4] = where(clamp, 0., grad_derivative_unorm);
    condition_grad[5] = where(clamp, 0., grad_derivative_plus_one_unorm);
}

template<typename T>
KERNELSPEC void kernel_rqs_inverse(
    FIn<T,0> input, FIn<T,1> condition, FOut<T,0> output, FOut<T,0> det
) {
    FVal<T> height(condition[0]), width(condition[1]);
    FVal<T> cumheight(condition[2]), cumwidth(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = (where(
        derivative_unorm > 20., derivative_unorm, log1p(exp(derivative_unorm))
    ) + MIN_DERIVATIVE) / softplus_scale;
    auto derivative_plus_one = (where(
        derivative_plus_one_unorm > 20.,
        derivative_plus_one_unorm,
        log1p(exp(derivative_plus_one_unorm))
    ) + MIN_DERIVATIVE) / softplus_scale;

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto delta = height / width;
    auto two_delta = 2. * delta;
    auto input_diff = input01 - cumheight;
    auto d_sum = derivative + derivative_plus_one - two_delta;
    auto tmp = input_diff * d_sum;
    auto a = tmp + height * (delta - derivative);
    auto b = height * derivative - tmp;
    auto c = delta * input_diff;
    auto discriminant = b * b + 4. * a * c;
    auto theta = 2. * c / (b + sqrt(discriminant));
    auto out = cumwidth + theta * width;
    output = where(clamp, input, out);

    auto one_minus_theta = 1. - theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto denominator = delta + d_sum * theta_one_minus_theta;
    auto derivative_numerator = delta * delta * (
        derivative_plus_one * theta * theta +
        two_delta * theta_one_minus_theta +
        derivative * one_minus_theta * one_minus_theta
    );
    det = where(clamp, 1., denominator * denominator / derivative_numerator);
}

template<typename T>
KERNELSPEC void backward_kernel_rqs_inverse(
    FIn<T,0> input, FIn<T,1> condition,
    FIn<T,0> output_grad, FIn<T,0> det_grad,
    FOut<T,0> input_grad, FOut<T,1> condition_grad
) {
    FVal<T> height(condition[0]), width(condition[1]);
    FVal<T> cumheight(condition[2]), cumwidth(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto expd = exp(derivative_unorm);
    auto expdpo = exp(derivative_plus_one_unorm);
    auto log1pd = log1p(expd);
    auto log1pdpo = log1p(expdpo);
    auto spd = where(derivative_unorm > 20., derivative_unorm, log1pd);
    auto spdpo = where(
        derivative_plus_one_unorm > 20., derivative_plus_one_unorm, log1pdpo
    );
    auto spd_md = spd + MIN_DERIVATIVE;
    auto spdpo_md = spdpo + MIN_DERIVATIVE;
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = spd_md / softplus_scale;
    auto derivative_plus_one = spdpo_md / softplus_scale;

    auto delta = height / width;

    auto tmp1 = derivative + derivative_plus_one;
    auto two_delta = 2. * delta;
    auto tmp2 = tmp1 - two_delta;
    auto input_diff = input01 - cumheight;
    auto tmp3 = input_diff * tmp2;
    auto tmp4 = delta - derivative;
    auto tmp5 = height * tmp4;
    auto a = tmp3 + tmp5;
    auto tmp6 = height * derivative;
    auto b = tmp6 - tmp3;
    auto c = delta * input_diff;
    auto tmp7 = b * b;
    auto tmp8 = 4. * a;
    auto tmp9 = tmp8 * c;
    auto discriminant = tmp7 + tmp9;
    auto tmp10 = 2. * c;
    auto tmp11 = sqrt(discriminant);
    auto tmp12 = b + tmp11;
    auto theta = tmp10 / tmp12;
    auto tmp13 = theta * width;
    auto out = tmp13 + cumwidth;

    auto one_minus_theta = 1. - theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto tmp14 = tmp2 * theta_one_minus_theta;
    auto denominator = delta + tmp14;
    auto tmp15 = delta * delta;
    auto tmp16 = derivative_plus_one * theta;
    auto tmp17 = tmp16 * theta;
    auto tmp18 = two_delta * theta_one_minus_theta;
    auto tmp19 = tmp17 + tmp18;
    auto tmp20 = derivative * one_minus_theta;
    auto tmp21 = tmp20 * one_minus_theta;
    auto tmp22 = tmp19 + tmp21;
    auto derivative_numerator = tmp15 * tmp22;
    auto tmp23 = denominator * denominator;
    auto tmp24 = tmp23 / derivative_numerator;

    auto grad_tmp24 = det_grad;
    auto grad_tmp23 = grad_tmp24 / derivative_numerator;
    auto grad_derivative_numerator = - grad_tmp24 * tmp23 / (
        derivative_numerator * derivative_numerator
    );
    auto grad_denominator = 2. * grad_tmp23 * denominator;

    auto grad_tmp15 = tmp22 * grad_derivative_numerator;
    auto grad_tmp22 = tmp15 * grad_derivative_numerator;
    auto grad_tmp19 = grad_tmp22;
    auto grad_tmp21 = grad_tmp22;
    auto grad_tmp20 = one_minus_theta * grad_tmp21;
    auto grad_one_minus_theta = tmp20 * grad_tmp21;
    auto grad_derivative = one_minus_theta * grad_tmp20;
    grad_one_minus_theta += derivative * grad_tmp20;
    auto grad_tmp17 = grad_tmp19;
    auto grad_tmp18 = grad_tmp19;
    auto grad_two_delta = theta_one_minus_theta * grad_tmp18;
    auto grad_theta_one_minus_theta = two_delta * grad_tmp18;
    auto grad_tmp16 = theta * grad_tmp17;
    auto grad_theta = tmp16 * grad_tmp17;
    auto grad_derivative_plus_one = theta * grad_tmp16;
    grad_theta += derivative_plus_one * grad_tmp16;
    auto grad_delta = 2. * delta * grad_tmp15;
    grad_delta += grad_denominator;
    auto grad_tmp14 = grad_denominator;
    auto grad_tmp2 = theta_one_minus_theta * grad_tmp14;
    grad_theta_one_minus_theta += tmp2 * grad_tmp14;
    grad_theta += one_minus_theta * grad_theta_one_minus_theta;
    grad_one_minus_theta += theta * grad_theta_one_minus_theta;
    grad_theta += - grad_one_minus_theta;

    auto grad_tmp13 = output_grad;
    auto grad_cumwidth = output_grad;
    auto grad_width = theta * grad_tmp13;
    grad_theta += width * grad_tmp13;
    auto grad_tmp10 = grad_theta / tmp12;
    auto grad_tmp12 = - tmp10 * grad_theta / (tmp12 * tmp12);
    auto grad_b = grad_tmp12;
    auto grad_tmp11 = grad_tmp12;
    auto grad_discriminant = 0.5 * grad_tmp11 / tmp11;
    auto grad_c = 2. * grad_tmp10;
    auto grad_tmp7 = grad_discriminant;
    auto grad_tmp9 = grad_discriminant;
    grad_c += tmp8 * grad_tmp9;
    auto grad_tmp8 = c * grad_tmp9;
    auto grad_a = 4. * grad_tmp8;
    grad_b += 2. * b * grad_tmp7;
    grad_delta += input_diff * grad_c;
    auto grad_input_diff = delta * grad_c;
    auto grad_tmp6 = grad_b;
    auto grad_tmp3 = -grad_b;
    auto grad_height = derivative * grad_tmp6;
    grad_derivative += height * grad_tmp6;
    grad_tmp3 += grad_a;
    auto grad_tmp5 = grad_a;
    grad_height += tmp4 * grad_tmp5;
    auto grad_tmp4 = height * grad_tmp5;
    grad_delta += grad_tmp4;
    grad_derivative += -grad_tmp4;
    grad_input_diff += tmp2 * grad_tmp3;
    grad_tmp2 += input_diff * grad_tmp3;
    auto grad_input = grad_input_diff;
    auto grad_cumheight = -grad_input_diff;
    auto grad_tmp1 = grad_tmp2;
    grad_two_delta += - grad_tmp2;
    grad_delta += 2. * grad_two_delta;
    grad_derivative += grad_tmp1;
    grad_derivative_plus_one += grad_tmp1;

    grad_height += grad_delta / width;
    grad_width += - height * grad_delta / (width * width);

    auto grad_spdpo_md = grad_derivative_plus_one / softplus_scale;
    auto grad_spd_md = grad_derivative / softplus_scale;
    auto grad_spdpo = grad_spdpo_md;
    auto grad_spd = grad_spd_md;

    auto grad_log1pdpo = grad_spdpo;
    auto grad_expdpo = grad_log1pdpo / (1. + expdpo);
    auto grad_derivative_plus_one_unorm = where(
        derivative_plus_one_unorm > 20., grad_spdpo, expdpo * grad_expdpo
    );

    auto grad_log1pd = grad_spd;
    auto grad_expd = grad_log1pd / (1. + expd);
    auto grad_derivative_unorm = where(derivative_unorm > 20., grad_spd, expd * grad_expd);

    input_grad = where(clamp, output_grad, grad_input);
    condition_grad[0] = where(clamp, 0., grad_height);
    condition_grad[1] = where(clamp, 0., grad_width);
    condition_grad[2] = where(clamp, 0., grad_cumheight);
    condition_grad[3] = where(clamp, 0., grad_cumwidth);
    condition_grad[4] = where(clamp, 0., grad_derivative_unorm);
    condition_grad[5] = where(clamp, 0., grad_derivative_plus_one_unorm);
}

template<typename T>
KERNELSPEC void kernel_softmax(FIn<T,1> input, FOut<T,1> output) {
    FVal<T> norm(0.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto exp_in = exp(input[i]);
        norm = norm + exp_in;
        output[i] = exp_in;
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = output[i] / norm;
    }
}

template<typename T>
KERNELSPEC void backward_kernel_softmax(
    FIn<T,1> output, FIn<T,1> output_grad, FOut<T,1> input_grad
) {
    std::size_t dim = output.size();
    FVal<T> grad_sum(0.);
    for (std::size_t i = 0; i < dim; ++i) {
        grad_sum = grad_sum + output[i] * output_grad[i];
    }
    for (std::size_t i = 0; i < dim; ++i) {
        input_grad[i] += output[i] * (output_grad[i] - grad_sum);
    }
}

template<typename T>
KERNELSPEC void kernel_softmax_prior(FIn<T,1> input, FIn<T,1> prior, FOut<T,1> output) {
    FVal<T> norm(0.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto unnorm_prob = exp(input[i]) * prior[i];
        norm = norm + unnorm_prob;
        output[i] = unnorm_prob;
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = output[i] / norm;
    }
}

template<typename T>
KERNELSPEC void backward_kernel_softmax_prior(
    FIn<T,1> output, FIn<T,1> output_grad, FOut<T,1> input_grad, FOut<T,1> prior_grad
) {
    //TODO: also gradient for prior?
    std::size_t dim = output.size();
    for (std::size_t i = 0; i < dim; ++i) {
        FVal<T> grad = output_grad[i];
        for (std::size_t j = 0; j < dim; ++j) {
            grad = grad - output[j] * output_grad[j];
        }
        input_grad[i] += output[i] * grad;
    }
}

template<typename T>
KERNELSPEC void kernel_select(FIn<T,1> input, IIn<T,1> indices, FOut<T,1> output) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        output[i] = input[single_index(indices[i])];
    }
}

template<typename T>
KERNELSPEC void backward_kernel_select(
    IIn<T,1> indices, FIn<T,1> output_grad, FOut<T,1> input_grad, FOut<T,1> indices_grad
) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        input_grad[single_index(indices[i])] = output_grad[i];
    }
}

}
}
