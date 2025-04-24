#pragma once

#include "madevent/cuda/tensor.h"

#define KERNELSPEC __device__ __forceinline__

namespace madevent {
namespace kernels {

template<int dim> using FViewIn = const CudaTensorView<double, dim>;
template<int dim> using FViewOut = CudaTensorView<double, dim>;

struct CudaTypes {
    template<int dim> using FIn = const cuda::CudaTensorView<double, dim>;
    template<int dim> using IIn = const cuda::CudaTensorView<int64_t, dim>;
    template<int dim> using BIn = const cuda::CudaTensorView<bool, dim>;
    template<int dim> using FOut = cuda::CudaTensorView<double, dim>;
    template<int dim> using IOut = cuda::CudaTensorView<int64_t, dim>;
    template<int dim> using BOut = cuda::CudaTensorView<bool, dim>;
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
inline __device__ std::size_t single_index(int64_t arg) {
    return arg;
}

}
}
