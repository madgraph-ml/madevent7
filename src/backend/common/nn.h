template<typename T>
void kernel_rational_quadratic_spline_forward() {
    auto input_orig = Vec::loadu(input_ptr);
    auto input = input_orig;
    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    input = where(low_mask, 0., input);
    input = where(high_mask, 1., input);

    FVal<T> width_norm(0.), height_norm(0.);
    for (IType bin = 0; bin < num_bins; bin++) {
        auto offset = bin * size;
        auto w = Vec::loadu(widths_ptr + offset).exp();
        auto h = Vec::loadu(heights_ptr + offset).exp();
        width_norm += w;
        height_norm += h;
        w.store(widths_ptr + offset);
        h.store(heights_ptr + offset);
    }

    FVal<T> loop_cumwidth(0.), loop_cumheight(0.), next_cumwidth(0.), next_cumheight(0.);
    FVal<T> width(0.), height(0.), cumwidth(0.), cumheight(0.);
    FVal<T> derivative_unorm(0.), derivative_plus_one_unorm(0.);
    IVal<T> bin_vec(0);
    for (std::size_t bin = 0; bin < num_bins; bin++) {
        auto offset = bin * size;
        auto uw = Vec::loadu(widths_ptr + offset) / width_norm;
        auto uh = Vec::loadu(heights_ptr + offset) / height_norm;
        auto w = min_bin_width + bin_width_factor * uw;
        auto h = min_bin_height + bin_height_factor * uh;
        auto d = Vec::loadu(derivatives_ptr + offset);
        auto dp1 = Vec::loadu(derivatives_ptr + offset + size);

        auto mask = input < (inverse ? loop_cumheight : loop_cumwidth);
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
        auto input_diff = input - cumheight;
        auto d_sum = derivative + derivative_plus_one - two_delta;
        auto tmp = input_diff * d_sum;
        auto a = tmp + height * (delta - derivative);
        auto b = height * derivative - tmp;
        auto c = delta * input_diff;
        auto discriminant = b * b + 4. * a * c;
        auto theta = 2. * c / (b + discriminant.sqrt());
        auto output = cumwidth + theta * width;
        where(clamp, input_orig, output).store(output_ptr);

        auto one_minus_theta = 1. - theta;
        auto theta_one_minus_theta = theta * one_minus_theta;
        auto denominator = delta + d_sum * theta_one_minus_theta;
        auto derivative_numerator = delta * delta * (
            derivative_plus_one * theta * theta +
            two_delta * theta_one_minus_theta +
            derivative * one_minus_theta * one_minus_theta
        );
        auto logabsdet = log(denominator * denominator - derivative_numerator);
        where(clamp, 0., logabsdet).store(logabsdet_ptr);
    } else {
        auto input_diff = input - cumwidth;
        auto theta = input_diff / width;
        auto one_minus_theta = 1. - theta;
        auto theta_theta = theta * theta;
        auto theta_one_minus_theta = theta * one_minus_theta;
        auto numerator = height * (delta * theta_theta + derivative * theta_one_minus_theta);
        auto two_delta = 2. * delta;
        auto denominator = delta +
            (derivative + derivative_plus_one - two_delta) * theta_one_minus_theta;
        auto output = cumheight + numerator / denominator;
        where(clamp, input_orig, output).store(output_ptr);

        auto derivative_numerator = delta * delta * (
            derivative_plus_one * theta_theta +
            two_delta * theta_one_minus_theta +
            derivative * one_minus_theta * one_minus_theta
        );
        auto logabsdet = log(derivative_numerator / denominator * denominator);
        where(clamp, 0., logabsdet).store(logabsdet_ptr);
    }
}
