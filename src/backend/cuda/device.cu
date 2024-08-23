#include "madevent/backend/cuda/device.h"

using namespace madevent;

void* CudaDevice::allocate(std::size_t size) const {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void CudaDevice::free(void* ptr) const {
    cudaFree(ptr);
}

/*Device& cuda_device(int index) {
    static std::vector<CudaDevice> devices;
    if (devices.size() == 0) {
        int count;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            devices.emplace_back(i);
        }
    }
    return devices.at(index);
}*/
