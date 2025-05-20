#pragma once

#include "tensor.h"
#include "simd.h"

#include <cmath>

#define KERNELSPEC

namespace madevent {
namespace kernels {

struct CpuTypes {
    template<int dim> using FIn = const TensorView<double, dim>;
    template<int dim> using IIn = const TensorView<int64_t, dim>;
    template<int dim> using FOut = TensorView<double, dim>;
    template<int dim> using IOut = TensorView<int64_t, dim>;
    using FVal = double;
    using IVal = int64_t;
    using BVal = bool;
};

inline double where(bool condition, double val_true, double val_false) {
    return condition ? val_true : val_false;
}
inline int64_t where(bool condition, int64_t val_true, int64_t val_false) {
    return condition ? val_true : val_false;
}
inline double min(double arg1, double arg2) {
    return arg1 < arg2 ? arg1 : arg2;
}
inline double max(double arg1, double arg2) {
    return arg1 > arg2 ? arg1 : arg2;
}
inline std::size_t single_index(int64_t arg) {
    return arg;
}

using std::sqrt;
using std::sin;
using std::cos;
using std::sinh;
using std::cosh;
using std::atan2;
using std::pow;
using std::fabs;
using std::log;
using std::log1p;
using std::tan;
using std::atan;
using std::exp;

using std::isnan;

#ifdef USE_SIMD
struct SimdTypes {
    template<int dim> using FIn = const cpu::VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IIn = const cpu::VectorizedTensorView<IVec, int64_t, dim, false>;
    template<int dim> using FOut = cpu::VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IOut = cpu::VectorizedTensorView<IVec, int64_t, dim, false>;
    using FVal = FVec;
    using IVal = IVec;
    using BVal = BVec;
};
#else // USE_SIMD
using SimdTypes = CpuTypes;
#endif // USE_SIMD

}
}
