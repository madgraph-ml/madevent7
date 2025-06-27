#include "device.h"
#include "tensor.h"
#include "../kernels/kernels.h"

using namespace madevent;
using namespace madevent::cuda;
using namespace madevent::kernels;

void* CudaDevice::allocate(std::size_t size) const {
    void* ptr;
    check_error(cudaMalloc(&ptr, size));
    return ptr;
}

void CudaDevice::free(void* ptr) const {
    check_error(cudaFree(ptr));
}

void CudaDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(cudaMemcpy(to, from, size, cudaMemcpyDefault));
}

void CudaDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_copy<CudaTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), AsyncCudaDevice(0)
    );
    check_error(cudaDeviceSynchronize());
}

void CudaDevice::tensor_zero(Tensor& tensor) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_zero<CudaTypes>, 1, 1>(
        {&tensor}, {&tensor}, tensor.size(0), AsyncCudaDevice(0)
    );
    check_error(cudaDeviceSynchronize());
}

void CudaDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<CudaTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), AsyncCudaDevice(0)
    );
}

void CudaDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(
        cudaMemcpy(target.data(), source.data(), source.byte_size(), cudaMemcpyDefault)
    );
}

void* AsyncCudaDevice::allocate(std::size_t size) const {
    void* ptr;
    check_error(cudaMallocAsync(&ptr, size, _stream));
    return ptr;
}

void AsyncCudaDevice::free(void* ptr) const {
    check_error(cudaFreeAsync(ptr, _stream));
}

void AsyncCudaDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(cudaMemcpyAsync(to, from, size, cudaMemcpyDefault, _stream));
}

void AsyncCudaDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_copy<CudaTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), *this
    );
}

void AsyncCudaDevice::tensor_zero(Tensor& tensor) const {
    //TODO: this only accidentally works for types other than double
    tensor_foreach_dynamic<kernel_zero<CudaTypes>, 1, 1>(
        {&tensor}, {&tensor}, tensor.size(0), *this
    );
}

void AsyncCudaDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<CudaTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), *this
    );
}

void AsyncCudaDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(cudaMemcpyAsync(
        target.data(), source.data(), source.byte_size(), cudaMemcpyDefault, _stream
    ));
}

extern "C" DevicePtr get_device() {
    return &CudaDevice::instance();
}

