#pragma once

#include <cmath>

#include "accessor.h"

#define KERNELSPEC static inline

namespace madevent {
namespace cpu {

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

#include "../common/kernels.h"

}
}
