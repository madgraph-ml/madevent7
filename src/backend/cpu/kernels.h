#pragma once

#include "tensor.h"
#include "simd.h"

#include <cmath>

#define KERNELSPEC

namespace {

struct CpuTypes {
    template<int dim> using FIn = const madevent::TensorView<double, dim>;
    template<int dim> using IIn = const madevent::TensorView<long long, dim>;
    template<int dim> using BIn = const madevent::TensorView<bool, dim>;
    template<int dim> using FOut = madevent::TensorView<double, dim>;
    template<int dim> using IOut = madevent::TensorView<long long, dim>;
    template<int dim> using BOut = madevent::TensorView<bool, dim>;
    using FVal = double;
    using IVal = long long;
    using BVal = bool;
};

template<typename T>
T where(bool condition, T val_true, T val_false) {
    return condition ? val_true : val_false;
}
std::size_t single_index(long long arg) {
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
using std::tan;
using std::atan;
using std::exp;

using std::isnan;

struct SimdTypes {
    template<int dim> using FIn = const VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IIn = const VectorizedTensorView<IVec, long long, dim, false>;
    template<int dim> using BIn = const VectorizedTensorView<BVec, bool, dim, false>;
    template<int dim> using FOut = VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IOut = VectorizedTensorView<IVec, long long, dim, false>;
    template<int dim> using BOut = VectorizedTensorView<BVec, bool, dim, false>;
    using FVal = FVec;
    using IVal = IVec;
    using BVal = BVec;
};

#include "../common/kernels.h"

}
