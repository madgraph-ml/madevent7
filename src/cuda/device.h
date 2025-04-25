#pragma once

#include "madevent/runtime/tensor.h"

#include <cuda_runtime.h>

namespace madevent {
namespace cuda {

class CudaDevice : public Device {
public:
    void* allocate(std::size_t size) const override;
    void free(void* ptr) const override;
    void memcpy(void* to, void* from, std::size_t size) const override;

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;
    void tensor_cpu(const Tensor& source, Tensor& target) const override;
    DevicePtr device_ptr() const override { return &instance(); }

    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(CudaDevice&) = delete;
    static const CudaDevice& instance() {
        static CudaDevice device;
        return device;
    }
private:
    CudaDevice() = default;
};

class AsyncCudaDevice {
public:
    AsyncCudaDevice(cudaStream_t stream) : _stream(stream) {}

    void* allocate(std::size_t size) const;
    void free(void* ptr) const;
    void memcpy(void* to, void* from, std::size_t size) const;

    void tensor_copy(const Tensor& source, Tensor& target) const;
    void tensor_zero(Tensor& tensor) const;
    void tensor_add(const Tensor& source, Tensor& target) const;
    void tensor_cpu(const Tensor& source, Tensor& target) const;
    DevicePtr device_ptr() const { return &CudaDevice::instance(); }
    cudaStream_t stream() const { return _stream; }

private:
    cudaStream_t _stream;
};

}
}
