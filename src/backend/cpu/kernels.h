#pragma once

#include "madevent/backend/tensor.h"

#include <cmath>

#define KERNELSPEC

namespace {

using std::sqrt;
using std::sin;
using std::cos;
using std::sinh;
using std::cosh;
using std::atan2;
using std::pow;
using std::fabs;
using std::log;
using std::tan;
using std::atan;
using std::exp;

template<int dim> using FViewIn = const madevent::TensorView<double, dim>;
template<int dim> using FViewOut = madevent::TensorView<double, dim>;

#include "../common/kernels.h"

}
