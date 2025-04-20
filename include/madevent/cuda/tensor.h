#pragma once

#include "madevent/runtime/tensor.h"

#include <type_traits>
#include <algorithm>

namespace madevent {
namespace cuda {

template<class T, int _dim>
class CudaTensorView {
public:
    using DType = T;
    static const int dim = _dim;

    __device__ CudaTensorView(uint8_t* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ const CudaTensorView<T, _dim-1> operator[](std::size_t index) const {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ CudaTensorView<T, _dim-1> operator[](std::size_t index) {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    __device__ operator typename std::conditional_t<_dim == 0, T, Nothing>() const {
        return *reinterpret_cast<T*>(_data);
    }

    template<int d = _dim, typename = std::enable_if_t<d == 0>>
    __device__ T operator=(T value) {
        *reinterpret_cast<T*>(_data) = value;
        return value;
    }

    __device__ CudaTensorView<T, _dim>& operator=(CudaTensorView<T, _dim>& value) = delete;
    __device__ std::size_t size() const { return _shape[0]; }

//private:
    uint8_t* _data;
    std::size_t* _stride;
    std::size_t* _shape;
};

template<class T, int _dim>
class PackedCudaTensorView {
public:
    using DType = T;
    static const int dim = _dim;

    PackedCudaTensorView(const madevent::TensorView<T, _dim>& view) :
        data(view.data())
    {
        if constexpr (_dim > 0) {
            std::copy(view.shape(), view.shape() + dim, std::begin(shape));
            std::copy(view.stride(), view.stride() + dim, std::begin(stride));
        }
    }

    /*__host__ __device__ CudaTensorView(
        uint8_t* _data, const std::size_t* _stride, const std::size_t* _shape
    ) : data(_data) {
        if constexpr (_dim > 0) {
            std::copy(_shape, _shape + dim, std::begin(shape));
            std::copy(_stride, _stride + dim, std::begin(stride));
        }
    }*/

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ const CudaTensorView<T, _dim-1> operator[](std::size_t index) const {
        return {data + index * stride[0], stride + 1, shape + 1};
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    __device__ CudaTensorView<T, _dim-1> operator[](std::size_t index) {
        return {data + index * stride[0], stride + 1, shape + 1};
    }

    __device__ operator typename std::conditional_t<_dim == 0, T, Nothing>() const {
        return *reinterpret_cast<T*>(data);
    }

    template<int d = _dim, typename = std::enable_if_t<d == 0>>
    __device__ T operator=(T value) {
        *reinterpret_cast<T*>(data) = value;
        return value;
    }

    __device__ CudaTensorView<T, _dim>& operator=(CudaTensorView<T, _dim>& value) = delete;
    __device__ std::size_t size() const { return shape[0]; }

//private:
    uint8_t* data;
    std::size_t stride[_dim];
    std::size_t shape[_dim];
};

}
}
