
inline constexpr double MIN_BIN_WIDTH = 1e-3;
inline constexpr double MIN_BIN_HEIGHT = 1e-3;
inline constexpr double MIN_DERIVATIVE = 1e-3;

template<typename T>
KERNELSPEC void rqs_forward_body(
    FIn<T,0> input, FIn<T,0> width, FIn<T,0> height, FIn<T,0> cumwidth,
    FIn<T,0> cumheight, FIn<T,0> derivative_unorm, FIn<T,0> derivative_plus_one_unorm,
    FOut<T,0> output, FOut<T,0> det
) {
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
    det = where(clamp, 1., derivative_numerator / denominator * denominator);
}

template<typename T>
KERNELSPEC void rqs_inverse_body(
    FIn<T,0> input, FIn<T,0> width, FIn<T,0> height, FIn<T,0> cumwidth,
    FIn<T,0> cumheight, FIn<T,0> derivative_unorm, FIn<T,0> derivative_plus_one_unorm,
    FOut<T,0> output, FOut<T,0> det
) {
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

// Kernels

template<typename T>
KERNELSPEC void kernel_leaky_relu(FIn<T,0> input, FOut<T,0> output) {
    FVal<T> x = input;
    output = where(x < 0, x * 0.01, x);
}

template<typename T>
KERNELSPEC void backward_kernel_leaky_relu(
    FIn<T,1> input, FIn<T,1> output_grad, FOut<T,1> input_grad
) {
    backward<kernel_leaky_relu<AutogradTypes>, 1, 1>(
        {input}, {output_grad}, {input_grad}
    );
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
        for (std::size_t i = 0; i < n_bins; ++i) {
            d_out[i] = input[offset + i];
        }
    }
}

template<typename T>
KERNELSPEC void backward_kernel_rqs_activation(
    FIn<T,2> widths, FIn<T,2> heights,
    FIn<T,2> widths_grad, FIn<T,2> heights_grad, FIn<T,2> derivatives_grad,
    FOut<T,1> input_grad
) {
    std::size_t n_dims = widths.size();
    std::size_t n_bins = (input_grad.size() / n_dims - 1) / 3;
    std::size_t n_cond = 3 * n_bins + 1;

    for (std::size_t j = 0; j < n_dims; ++j) {
        auto w_j = widths[j];
        auto grad_w_j = widths_grad[j];
        std::size_t offset = j * n_cond;
        for (std::size_t i = 0; i < n_bins; ++i) {
            FVal<T> grad = grad_w_j[i];
            for (std::size_t k = 0; j < n_bins; ++k) {
                grad = grad - w_j[j] * grad_w_j[j];
            }
            input_grad[offset+i] = w_j[i] * grad;
        }

        auto h_j = heights[j];
        auto grad_h_j = heights_grad[j];
        offset += n_bins;
        for (std::size_t i = 0; i < n_bins; ++i) {
            FVal<T> grad = grad_h_j[i];
            for (std::size_t k = n_bins; j < 2*n_bins; ++k) {
                grad = grad - h_j[j] * grad_h_j[j];
            }
            input_grad[offset+i] = h_j[i] * grad;
        }

        offset += n_bins;
        for (std::size_t i = 0; i < n_bins; ++i) {
            input_grad[offset+i] = derivatives_grad[j][i];
        }
    }
}

template<typename T>
KERNELSPEC void kernel_rqs_find_bin(
    FIn<T,1> input, FIn<T,2> in_sizes, FIn<T,2> out_sizes, FIn<T,2> derivatives,
    FOut<T,2> condition
) {
    /*FVal<T> det_product(1.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto condition_i = condition[i];
        auto n_bins = condition_i.size();
        auto input_orig = input[0];
        auto low_mask = input_orig < 0.;
        auto high_mask = input_orig > 1.;
        auto clamp = low_mask | high_mask;
        auto input01 = where(high_mask, 1., where(low_mask, 0., input_orig));

        FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
        FVal<T> width(0.), height(0.), cumwidth(0.), cumheight(0.);
        FVal<T> derivative_unorm(0.), derivative_plus_one_unorm(0.);
        for (std::size_t bin = 0; bin < n_bins; ++bin) {
            auto w = min_bin_width + bin_width_factor * condition_i[bin];
            auto h = min_bin_height + bin_height_factor * condition_i[n_bins + bin];
            auto d = condition_i[2 * n_bins + bin];
            auto dp1 = condition_i[2 * n_bins + bin + 1];

            auto mask = input01 < (inverse ? loop_cumheight : loop_cumwidth);
            width = where(mask, width, w);
            height = where(mask, height, h);
            derivative_unorm = where(mask, derivative_unorm, d);
            derivative_plus_one_unorm = where(mask, derivative_plus_one_unorm, dp1);
            cumwidth = where(mask, cumwidth, loop_cumwidth);
            cumheight = where(mask, cumheight, loop_cumheight);
            loop_cumwidth += w;
            loop_cumheight += h;
        }



    }
    det = det_product;*/
}

template<typename T>
KERNELSPEC void kernel_rqs_forward(
    FIn<T,1> input, FIn<T,2> condition, FOut<T,1> output, FOut<T,1> det
) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto condition_i = condition[i];
        rqs_forward_body<T>(
            input[i],
            condition_i[0],
            condition_i[1],
            condition_i[2],
            condition_i[3],
            condition_i[4],
            condition_i[5],
            output[i], 
            det[i]
        );
    }
}

/*template<typename T>
KERNELSPEC void backward_kernel_rqs_forward(
    FIn<T,1> input, FIn<T,2> condition, FIn<T,1> det,
    FIn<T,1> output_grad, FIn<T,0> det_grad,
    FOut<T,1> input_grad, FOut<T,2> condition_grad
) {
    FVal<T> det_product(1.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto condition_i = condition[i];
        auto condition_grad_i = condition_grad[i];
        FVal<T> det_grad_i = ;
        backward<rqs_forward_body<AutogradTypes>, 7, 2>(
            {
                input[i],
                condition_i[0],
                condition_i[1],
                condition_i[2],
                condition_i[3],
                condition_i[4],
                condition_i[5],
            },
            {
                output_grad,
                det_grad,
            },
            {
                input_grad[i],
                condition_grad_i[0],
                condition_grad_i[1],
                condition_grad_i[2],
                condition_grad_i[3],
                condition_grad_i[4],
                condition_grad_i[5],
            }
        );
        det_product = det_product * det_i;
    }
    det = det_product;
}*/

template<typename T>
KERNELSPEC void kernel_rqs_inverse(
    FIn<T,1> input, FIn<T,2> condition, FOut<T,1> output, FOut<T,1> det
) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto condition_i = condition[i];
        rqs_inverse_body<T>(
            input[i],
            condition_i[0],
            condition_i[1],
            condition_i[2],
            condition_i[3],
            condition_i[4],
            condition_i[5],
            output[i], 
            det[i]
        );
    }
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
    for (std::size_t i = 0; i < dim; ++i) {
        FVal<T> grad = output_grad[i];
        for (std::size_t j = 0; j < dim; ++j) {
            grad = grad - output[j] * output_grad[j];
        }
        input_grad[i] = output[i] * grad;
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
    FIn<T,1> output, FIn<T,1> output_grad, FOut<T,1> input_grad
) {
    //TODO: also gradient for prior?
    std::size_t dim = output.size();
    for (std::size_t i = 0; i < dim; ++i) {
        FVal<T> grad = output_grad[i];
        for (std::size_t j = 0; j < dim; ++j) {
            grad = grad - output[j] * output_grad[j];
        }
        input_grad[i] = output[i] * grad;
    }
}
