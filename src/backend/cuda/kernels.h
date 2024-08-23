#pragma once

#include "madevent/backend/cuda/tensor.h"

#define KERNELSPEC __device__ __forceinline__

using namespace madevent::cuda;

namespace madevent {
namespace cuda {

using DoubleInput = const CudaTensorView<double>;
using DoubleOutput = CudaTensorView<double>;

#include "../common/kernels.h"

}
}
