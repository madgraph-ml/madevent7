// Helper functions

/*template<typename T>
void _rational_quadratic_spline(
    FIn<T,1> input, FIn<T,2> condition, FOut<T,1> output, FOut<T,0> log_det
    FOut<T,1> tmp_exp_w, FOut<T,1> tmp_exp_h, bool inverse
) {
    FVal<T> log_det_sum(0.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto condition_i = condition[i];
        auto n_bins = condition_i.size();
        auto input_orig = input[0];
        auto low_mask = input_orig < 0.;
        auto high_mask = input_orig > 1.;
        auto clamp = low_mask | high_mask;
        auto input01 = where(high_mask, 1., where(low_mask, 0., input_orig));

        FVal<T> width_norm(0.), height_norm(0.);
        for (std::size_t bin = 0; bin < n_bins; ++bin) {
            auto offset = bin * size;
            auto w = exp(condition_i[bin]);
            auto h = exp(condition_i[n_bins + bin]);
            width_norm += w;
            height_norm += h;
            tmp_exp_w[bin] = w;
            tmp_exp_h[bin] = h;
        }

        FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
        FVal<T> width(0.), height(0.), cumwidth(0.), cumheight(0.);
        FVal<T> derivative_unorm(0.), derivative_plus_one_unorm(0.);
        IVal<T> bin_vec(0);
        for (std::size_t bin = 0; bin < n_bins; ++bin) {
            auto offset = bin * size;
            auto uw = tmp_exp_w[bin] / width_norm;
            auto uh = tmp_exp_h[bin] / height_norm;
            auto w = min_bin_width + bin_width_factor * uw;
            auto h = min_bin_height + bin_height_factor * uh;
            auto d = condition_i[2 * n_bins + bin];
            auto dp1 = condition_i[2 * n_bins + bin + 1];

            auto mask = input01 < (inverse ? loop_cumheight : loop_cumwidth);
            width = where(mask, width, w);
            height = where(mask, height, h);
            derivative_unorm = where(mask, derivative_unorm, d);
            derivative_plus_one_unorm = where(mask, derivative_plus_one_unorm, dp1);
            cumwidth = where(mask, cumwidth, loop_cumwidth);
            cumheight = where(mask, cumheight, loop_cumheight);
            bin_vec = where(mask, bin_vec, bin);

            loop_cumwidth += w;
            loop_cumheight += h;
        }

        auto softplus_scale = LOG_TWO + min_derivative;
        auto derivative = (where(
            derivative_unorm > 20., derivative_unorm, log1p(exp(derivative_unorm))
        ) + min_derivative) / softplus_scale;
        auto derivative_plus_one = (where(
            derivative_plus_one_unorm > 20.,
            derivative_plus_one_unorm,
            log1p(exp(derivative_plus_one_unorm))
        ) + min_derivative) / softplus_scale;

        auto delta = height / width;

        if (inverse) {
            auto two_delta = 2. * delta;
            auto input_diff = input01 - cumheight;
            auto d_sum = derivative + derivative_plus_one - two_delta;
            auto tmp = input_diff * d_sum;
            auto a = tmp + height * (delta - derivative);
            auto b = height * derivative - tmp;
            auto c = delta * input_diff;
            auto discriminant = b * b + 4. * a * c;
            auto theta = 2. * c / (b + discriminant.sqrt());
            auto output = cumwidth + theta * width;
            output = where(clamp, input_orig, output);

            auto one_minus_theta = 1. - theta;
            auto theta_one_minus_theta = theta * one_minus_theta;
            auto denominator = delta + d_sum * theta_one_minus_theta;
            auto derivative_numerator = delta * delta * (
                derivative_plus_one * theta * theta +
                two_delta * theta_one_minus_theta +
                derivative * one_minus_theta * one_minus_theta
            );
            log_det_sum = log_det_sum +
                where(clamp, 0., log(denominator * denominator - derivative_numerator));
        } else {
            auto input_diff = input01 - cumwidth;
            auto theta = input_diff / width;
            auto one_minus_theta = 1. - theta;
            auto theta_theta = theta * theta;
            auto theta_one_minus_theta = theta * one_minus_theta;
            auto numerator = height * (delta * theta_theta + derivative * theta_one_minus_theta);
            auto two_delta = 2. * delta;
            auto denominator = delta +
                (derivative + derivative_plus_one - two_delta) * theta_one_minus_theta;
            auto output = cumheight + numerator / denominator;
            output = where(clamp, input_orig, output);

            auto derivative_numerator = delta * delta * (
                derivative_plus_one * theta_theta +
                two_delta * theta_one_minus_theta +
                derivative * one_minus_theta * one_minus_theta
            );
            log_det_sum = log_det_sum +
                where(clamp, 0., log(derivative_numerator / denominator * denominator));
        }
    }
    log_det = log_det_sum;
}*/


// Kernels

template<typename T>
KERNELSPEC void kernel_leaky_relu(FIn<T,1> input, FOut<T,1> output) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        FVal<T> x = input[i];
        output[i] = where(x < 0, 0.01 * x, x);
    }
}

template<typename T>
KERNELSPEC void kernel_rqs_activation(
    FIn<T,1> input, IIn<T,0> bin_count,
    FOut<T,2> output, FOut<T,1> w_norms, FOut<T,1> h_norms
) {
    std::size_t n_bins = single_index(bin_count);
    std::size_t n_cond = 3 * n_bins + 1;
    std::size_t n_dims = input.size() / n_cond;

    for (std::size_t j = 0; j < n_bins; ++j) {
        auto out_dim = output[j];
        FVal<T> w_norm(0.);
        FVal<T> h_norm(0.);
        std::size_t offset = j * n_cond;
        std::size_t i = 0;
        for (; i < n_bins; ++i) {
            auto w = exp(input[offset + i]);
            w_norm = w_norm + w;
            out_dim[i] = w;
        }
        for (; i < 2 * n_bins; ++i) {
            auto h = exp(input[offset + i]);
            h_norm = h_norm + h;
            out_dim[i] = h;
        }
        for (; i < 3 * n_bins + 1; ++i) {
            out_dim[i] = input[offset + i];
        }
        w_norms[j] = w_norm;
        h_norms[j] = h_norm;
    }
}

template<typename T>
KERNELSPEC void kernel_rqs_forward(
    FIn<T,1> input, FIn<T,2> condition, FIn<T,1> w_norms, FIn<T,1> h_norms,
    FOut<T,1> output, FOut<T,0> det
) {
    //_rational_quadratic_spline(input, condition, output, log_det, tmp_exp_w, tmp_exp_h, false);
}

template<typename T>
KERNELSPEC void kernel_rqs_inverse(
    FIn<T,1> input, FIn<T,2> condition, FIn<T,1> w_norms, FIn<T,1> h_norms,
    FOut<T,1> output, FOut<T,0> det
) {
    //_rational_quadratic_spline(input, condition, output, log_det, tmp_exp_w, tmp_exp_h, true);
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
