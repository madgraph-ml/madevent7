#pragma once

#include <string>

namespace madevent {

std::string format_si_prefix(double value);
std::string format_with_error(double value, double error);
std::string format_progress(double progress, int width);

}
