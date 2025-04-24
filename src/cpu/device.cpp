#include "device.h"
#include "tensor.h"
#include "../kernels/kernels.h"

using namespace madevent;
using namespace madevent::cpu;
using namespace madevent::kernels;

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
        {&source}, {&target}, target.size(0)
    );
}

void CpuDevice::tensor_zero(Tensor& tensor) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_zero<CpuTypes>, kernel_zero<SimdTypes>, 1, 1>(
        {&tensor}, {&tensor}, tensor.size(0)
    );
}

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<CpuTypes>, kernel_add_inplace<SimdTypes>, 1, 1>(
        {&source}, {&target}, target.size(0)
    );
}

extern "C" DevicePtr get_device() {
    return &CpuDevice::instance();
}
