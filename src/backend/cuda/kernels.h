#pragma once

#include "madevent/backend/cpu/tensor.h"

#define KERNELSPEC __device__ __forceinline__

using namespace madevent::cuda;

namespace madevent {
namespace cuda {

using DoubleInput = const TensorView<double>;
using DoubleOutput = TensorView<double>;

#include "../common/kernels.h"

}
}
