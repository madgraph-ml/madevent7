#pragma once

#include "tensor.h"
#include <cuda_runtime.h>

#define KERNELSPEC __device__ //__forceinline__

namespace madevent {
namespace kernels {

struct CudaTypes {
    template<int dim> using FIn = const cuda::CudaTensorView<double, dim>;
    template<int dim> using IIn = const cuda::CudaTensorView<int64_t, dim>;
    template<int dim> using FOut = cuda::CudaTensorView<double, dim>;
    template<int dim> using IOut = cuda::CudaTensorView<int64_t, dim>;
    using FVal = double;
    using IVal = int64_t;
    using BVal = bool;
};

inline __device__ double where(bool condition, double val_true, double val_false) {
    return condition ? val_true : val_false;
}
inline __device__ int64_t where(bool condition, int64_t val_true, int64_t val_false) {
    return condition ? val_true : val_false;
}
inline __device__ double min(double arg1, double arg2) {
    return arg1 < arg2 ? arg1 : arg2;
}
inline __device__ double max(double arg1, double arg2) {
    return arg1 > arg2 ? arg1 : arg2;
}
inline __device__ std::size_t single_index(int64_t arg) {
    return arg;
}

using ::sqrt;
using ::sin;
using ::cos;
using ::sinh;
using ::cosh;
using ::atan2;
using ::atanh;
using ::pow;
using ::fabs;
using ::log;
using ::log1p;
using ::tan;
using ::atan;
using ::exp;

using ::isnan;

}
}
