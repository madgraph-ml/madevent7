#pragma once

#include "madevent/runtime/tensor.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <format>

namespace madevent {
namespace cuda {

inline void check_error(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *error_str = cublasGetStatusString(status);
        throw std::runtime_error(std::format("CUBLAS error: {}", error_str));
    }
}

inline void check_error(curandStatus_t status) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::format("CURAND error: error code {}", static_cast<int>(status))
        );
    }
}

inline void check_error(cudaError_t error) {
    if (error != cudaSuccess) {
        const char *error_str = cudaGetErrorString(error);
        throw std::runtime_error(std::format("CUDA error: {}", error_str));
    }
}

inline void check_error() {
    check_error(cudaGetLastError());
}

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

extern "C" DevicePtr get_device();

}
}
