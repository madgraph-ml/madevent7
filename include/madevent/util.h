#pragma once

#include <string>

namespace madevent {

template<class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

std::string format_si_prefix(double value);
std::string format_with_error(double value, double error);
std::string format_progress(double progress, int width);

}
