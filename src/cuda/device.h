#pragma once

#include "madevent/runtime/tensor.h"

#include <vector>

namespace madevent {
namespace cuda {

class CudaDevice : public Device {
public:
    void* allocate(std::size_t size) const override;
    void free(void* ptr) const override;

    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(CudaDevice&) = delete;
    friend inline CudaDevice& cuda_device();
private:
    CudaDevice() {}
};

inline CudaDevice& cuda_device() {
    static CudaDevice inst;
    return inst;
}

}
}
