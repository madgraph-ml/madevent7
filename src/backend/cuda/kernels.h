#pragma once

#include "madevent/backend/cuda/tensor.h"

#define KERNELSPEC __device__ __forceinline__

using namespace madevent::cuda;

namespace madevent {
namespace cuda {

template<int dim> using FViewIn = const CudaTensorView<double, dim>;
template<int dim> using FViewOut = CudaTensorView<double, dim>;

#include "../common/kernels.h"

}
}
