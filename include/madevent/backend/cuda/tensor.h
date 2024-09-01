#pragma once

#include "madevent/backend/tensor.h"

namespace madevent {
namespace cuda {

template<class T>
class CudaTensorView {
public:
    using DType = T;

    CudaTensorView(const madevent::TensorView<T>& view) :
        data(view.data()), stride(view.stride()), shape(view.stride()) {}
    __host__ __device__ CudaTensorView(uint8_t* _data, std::size_t* _stride, std::size_t* _shape) :
        data(_data), stride(_stride), shape(_shape) {}
    __device__ const CudaTensorView<T> operator[](std::size_t index) const {
        return CudaTensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    __device__ CudaTensorView<T> operator[](std::size_t index) {
        return CudaTensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    __device__ operator T() const { return *reinterpret_cast<T*>(data); }
    __device__ T operator=(T value) { *reinterpret_cast<T*>(data) = value; return value; }
    __device__ CudaTensorView<T>& operator=(CudaTensorView<T>& value) = delete;
    __device__ std::size_t size() const { return shape[0]; }

//private:
    uint8_t* data;
    std::size_t* stride;
    std::size_t* shape;
};

}
}
