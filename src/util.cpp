#include "madevent/util.h"

#include <cmath>
#include <format>
#include <sstream>

using namespace madevent;

namespace {

const std::array<std::string, 5> si_prefixes {"", "k", "M", "G", "T"};
const std::array<std::string, 9> progress_symbols {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};

}

std::string madevent::format_si_prefix(double value) {
    value = std::round(value);
    int value_power = std::floor(std::log10(value));
    int value_power3 = value_power / 3;

    if (value_power3 >= 0 && value_power3 < si_prefixes.size()) {
        int digits_after_dot = value_power3 == 0 ? 0 : 2 - value_power % 3;
        double value_scaled = value / std::pow(10, 3 * value_power3);
        return std::format(
            "{:.{}f}{}", value_scaled, digits_after_dot, si_prefixes.at(value_power3)
        );
    } else {
        return std::format("{}", value);
    }
}

std::string madevent::format_with_error(double value, double error) {
    int sig_power = 1 - int(std::floor(std::log10(error)));
    int value_power = std::floor(std::log10(value));
    if (sig_power < 0 || sig_power > 5) {
        std::string exp_fmt = std::format("{:.{}e}", value, value_power + sig_power);
        auto e_pos = exp_fmt.find("e");
        double err_val = error * std::pow(10, sig_power);
        return std::format(
            "{}({:.0f})e{}", exp_fmt.substr(0, e_pos), err_val, exp_fmt.substr(e_pos + 1)
        );
    } else {
        int err_prec = sig_power == 1;
        double err_val = error * std::pow(10, sig_power - err_prec);
        return std::format("{:.{}f}({:.{}f})", value, sig_power, err_val, err_prec);
    }
}

std::string madevent::format_progress(double progress, int width) {
    double frac = width * std::min(1.0, std::max(0.0, progress));
    int n_full = frac;
    std::stringstream str;
    for (int i = 0; i < n_full; ++i) str << progress_symbols.back();
    int n_remaining;
    if (n_full >= width) {
        n_remaining = 0;
    } else {
        str << progress_symbols.at(int((frac - n_full) * progress_symbols.size()));
        n_remaining = width - n_full - 1;
    }
    for (int i = 0; i < n_remaining; ++i) str << progress_symbols.front();
    return str.str();
}
