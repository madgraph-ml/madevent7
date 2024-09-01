#pragma once

#include "madevent/madcode/function.h"

#include <cstdint>
#include <vector>
#include <array>
#include <functional>
#include <type_traits>

namespace madevent {

using SizeVec = std::vector<std::size_t>;

template<class T, int _dim>
class TensorView {
public:
    using DType = T;
    static const int dim = _dim;

    TensorView(uint8_t* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    const TensorView<T, _dim-1> operator[](std::size_t index) const {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    TensorView<T, _dim-1> operator[](std::size_t index) {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    operator typename std::conditional_t<_dim == 0, T, void>() const {
        return *reinterpret_cast<T*>(_data);
    }

    template<int d = _dim, typename = std::enable_if_t<d == 0>>
    T operator=(T value) {
        *reinterpret_cast<T*>(_data) = value;
        return value;
    }

    TensorView<T, _dim>& operator=(TensorView<T, _dim>& value) = delete;
    std::size_t size() const { return _shape[0]; }
    uint8_t* data() const { return _data; }
    std::size_t* stride() const { return _stride; }
    std::size_t* shape() const { return _shape; }

private:
    uint8_t* _data;
    std::size_t* _stride;
    std::size_t* _shape;
};

class Device {
public:
    virtual void* allocate(std::size_t size) const = 0;
    virtual void free(void* ptr) const = 0;
};

class CpuDevice : public Device {
public:
    void* allocate(std::size_t size) const override {
        return new uint8_t[size];
    }

    void free(void* ptr) const override {
        delete[] static_cast<uint8_t*>(ptr);
    }

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    friend inline CpuDevice& cpu_device();

private:
    CpuDevice() {}
};

inline CpuDevice& cpu_device() {
    static CpuDevice inst;
    return inst;
}

class Tensor {
public:
    Tensor() = default;

    Tensor(const Tensor& other) : impl(other.impl) {
        if (impl != nullptr) ++impl->ref_count;
    }

    Tensor(Tensor&& other) : impl(other.impl) {
        other.impl = nullptr;
    }

    Tensor(DataType dtype, SizeVec shape) :
        Tensor(dtype, shape, cpu_device()) {}

    Tensor(DataType dtype, SizeVec shape, Device& device) :
        impl(new TensorImpl{dtype, shape, device})
    {
        auto size = init_stride();
        impl->data = device.allocate(size);
    }

    Tensor(DataType dtype, SizeVec shape, std::function<void*(std::size_t)> allocator) :
        Tensor(dtype, shape, cpu_device(), allocator) {}

    Tensor(
        DataType dtype, SizeVec shape, Device& device, std::function<void*(std::size_t)> allocator
    ) : impl(new TensorImpl{dtype, shape, device})
    {
        auto size = init_stride();
        impl->data = allocator(size);
    }

    Tensor(DataType dtype, SizeVec shape, void* data) :
        Tensor(dtype, shape, cpu_device(), data) {}

    Tensor(DataType dtype, SizeVec shape, Device& device, void* data) :
        impl(new TensorImpl{dtype, shape, device, data, false})
    {
        init_stride();
    }

    ~Tensor() {
        reset();
    }

    Tensor& operator=(const Tensor& other) {
        reset();
        impl = other.impl;
        if (impl != nullptr) ++impl->ref_count;
        return *this;
    }

    Tensor& operator=(Tensor&& other) {
        reset();
        impl = other.impl;
        other.impl = nullptr;
        return *this;
    }

    operator bool() {
        return impl != nullptr;
    }

    template<class T, int dim> TensorView<T, dim> view(bool flatten = false) {
        return TensorView<T, dim>(
            static_cast<uint8_t*>(impl->data) + impl->offset,
            impl->stride.data(),
            flatten ? impl->flat_shape.data() : impl->shape.data()
        );
    }

    void* data() {
        return impl->data;
    }

    SizeVec& shape() {
        return impl->shape;
    }

    SizeVec& stride() {
        return impl->stride;
    }

    std::size_t size(std::size_t i) {
        return impl->shape[i];
    }

    void reset() {
        if (impl == nullptr) return;
        impl->reset([this] (void* ptr) { impl->device.free(ptr); });
        impl = nullptr;
    }

    void reset(std::function<void(void*)> deleter) {
        if (impl == nullptr) return;
        impl->reset(deleter);
        impl = nullptr;
    }

private:
    struct TensorImpl {
        DataType dtype;
        SizeVec shape;
        Device& device;
        void* data;
        bool owns_data = true;
        TensorImpl* data_owner;
        int ref_count = 1;
        SizeVec stride;
        std::array<std::size_t, 2> flat_shape;
        std::size_t offset;

        void reset(std::function<void(void*)> deleter) {
            if (ref_count > 1) {
                --ref_count;
                return;
            }
            if (owns_data) {
                deleter(data);
            } else if (data_owner != nullptr) {
                data_owner->reset(deleter);
            }
            delete this;
        }
    };

    std::size_t init_stride();
    TensorImpl* impl;
};


}
