#include "device.h"
#include "tensor.h"
#include "../kernels/kernels.h"

using namespace madevent;
using namespace madevent::cpu;
using namespace madevent::kernels;

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    if (source.dtype() == DataType::dt_float && target.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else if (source.dtype() == DataType::dt_int && target.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_copy_int<CpuTypes>, kernel_copy_int<SimdTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in copy");
    }
}

void CpuDevice::tensor_zero(Tensor& tensor) const {
    if (tensor.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_zero<CpuTypes>, kernel_zero<SimdTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else if (tensor.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_zero_int<CpuTypes>, kernel_zero_int<SimdTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in zero");
    }
}

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<
        kernel_add_inplace<CpuTypes>, kernel_add_inplace<SimdTypes>, 1, 1
    >(
        {&source}, {&target}, target.size(0), *this
    );
}

extern "C" DevicePtr get_device() {
    return &CpuDevice::instance();
}
