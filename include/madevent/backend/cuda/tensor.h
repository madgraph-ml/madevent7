#pragma once

#include "madevent/backend/tensor.h"

#include <type_traits>
#include <algorithm>

namespace madevent {
namespace cuda {

template<class T, int _dim>
class CudaTensorView {
public:
    using DType = T;
    static const int dim = _dim;

    CudaTensorView(const madevent::TensorView<T>& view) :
        data(view.data()), stride(view.stride()), shape(view.stride()) {}

    __host__ __device__ CudaTensorView(
        uint8_t* _data, std::size_t* _stride, std::size_t* _shape
    ) : data(_data) {
        std::copy(_shape, _shape + dim, std::begin(shape));
        std::copy(_stride, _stride + dim, std::begin(stride));
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ const CudaTensorView<T, _dim-1> operator[](std::size_t index) const {
        return {data + index * stride[0], stride + 1, shape + 1};
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ CudaTensorView<T> operator[](std::size_t index) {
        return {data + index * stride[0], stride + 1, shape + 1};
    }

    __device__ operator typename std::conditional_t<_dim == 0, T, void>() const {
        return *reinterpret_cast<T*>(_data);
    }

    __device__ template<int d = _dim, typename = std::enable_if_t<d == 0>>
    T operator=(T value) {
        *reinterpret_cast<T*>(_data) = value;
        return value;
    }

    __device__ CudaTensorView<T>& operator=(CudaTensorView<T>& value) = delete;
    __device__ std::size_t size() const { return shape[0]; }

private:
    uint8_t* data;
    std::size_t[_dim] stride;
    std::size_t[_dim] shape;
};

}
}
