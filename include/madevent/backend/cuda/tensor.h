#pragma once

#include "madevent/madcode/function.h"

#include <cstdint>
#include <vector>
#include <array>
#include <memory>

namespace madevent {
namespace cuda {

using SizeVec = std::vector<std::size_t>;

template<class T>
class TensorView {
public:
    using DType = T;

    __host__ __device__ TensorView(uint8_t* _data, std::size_t* _stride, std::size_t* _shape) :
        data(_data), stride(_stride), shape(_shape) {}
    __device__ const TensorView<T> operator[](std::size_t index) const {
        return TensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    __device__ TensorView<T> operator[](std::size_t index) {
        return TensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    __device__ operator T() const { return *static_cast<T* const>(data); }
    __device__ T operator=(T value) { *static_cast<T*>(data) = value; return value; }
    __device__ TensorView<T>& operator=(TensorView<T>& value) = delete;
    __device__ std::size_t size() const { return shape[0]; }

private:
    void* data;
    std::size_t* stride;
    std::size_t* shape;
};


class Tensor {
public:
    Tensor() = default;

    Tensor(Tensor& other) : impl(other.impl) {
        if (impl.ref_count != nullptr) ++(*impl.ref_count);
    }

    Tensor(Tensor&& other) : impl(other.impl) {
        other.data = nullptr;
        other.ref_count = nullptr;
    }

    Tensor::Tensor(DataType dtype, SizeVec shape) : impl{dtype, shape, new int(1)} {
        cudaMalloc(&impl.data, init_stride());
    }

    Tensor::Tensor(DataType dtype, SizeVec shape, cudaStream_t stream) :
        impl{dtype, shape, new int(1)}
    {
        cudaMallocAsync(&impl.data, init_stride(), stream);
    }

    Tensor::Tensor(DataType _dtype, SizeVec _shape, void* _data) :
        dtype(_dtype), shape(_shape), data(_data)
    {
        init_stride();
    }

    Tensor::~Tensor() {
        reset();
    }

    Tensor& operator=(Tensor& other) {
        reset();
        impl = other.impl;
        if (impl.ref_count != nullptr) ++(*impl.ref_count);
        return *this;
    }

    Tensor& operator=(Tensor&& other) {
        reset();
        other.data = nullptr;
        other.ref_count = nullptr;
        impl = other.impl;
        return *this;
    }

    operator bool() {
        return impl.ref_count != nullptr;
    }

    template<class T> TensorView<T, true> view(bool flatten = false) {
        return TensorView<T>(
            impl.data + impl.offset,
            impl.stride.data(),
            flatten ? impl.flat_shape.data() : impl.shape.data()
        );
    }
    std::size_t size(std::size_t i) { return impl.shape[i]; }
    void reset();
    void reset_async(cudaStream_t stream);
    void* data() { return impl.data; }

private:
    std::size_t init_stride();

    struct {
        DataType dtype;
        SizeVec shape;
        int* ref_count;
        void* data;
        SizeVec stride;
        std::array<std::size_t, 2> flat_shape;
        std::size_t offset;
    } impl;
};

}
}
