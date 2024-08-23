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

using DoubleInput = const madevent::TensorView<double>;
using DoubleOutput = madevent::TensorView<double>;

#include "../common/kernels.h"

}
