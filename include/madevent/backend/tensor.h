#pragma once

#include "madevent/madcode/type.h"

#include <cstdint>
#include <vector>
#include <functional>
#include <type_traits>
#include <initializer_list>

namespace madevent {

struct Nothing;

using SizeVec = std::vector<std::size_t>;

class Sizes {
public:
    static constexpr std::size_t max_size = 4;

    Sizes() : _size(0) {};
    Sizes(std::size_t size) : _size(size) {
        std::fill(begin(), end(), 0);
    };
    Sizes(const SizeVec& values) : _size(values.size()) {
        if (values.size() > max_size) {
            throw std::invalid_argument("maximum dimension exceeded");
        }
        std::copy(values.begin(), values.end(), begin());
    }
    Sizes(std::initializer_list<std::size_t> values) : _size(values.size()) {
        if (values.size() > max_size) {
            throw std::invalid_argument("maximum dimension exceeded");
        }
        std::copy(values.begin(), values.end(), begin());
    }
    std::size_t& operator[](std::size_t index) { return _values[index]; }
    const std::size_t& operator[](std::size_t index) const { return _values[index]; }
    std::size_t size() const { return _size; }
    std::size_t* begin() { return &_values[0]; }
    std::size_t* end() { return &_values[_size]; }
    const std::size_t* begin() const { return &_values[0]; }
    const std::size_t* end() const { return &_values[_size]; }
    void push_back(std::size_t item) { _values[_size] = item; ++_size; }
    std::size_t* data() { return &_values[0]; }
    const std::size_t* data() const { return &_values[0]; }
    std::size_t& back() { return _values[_size - 1]; }
    const std::size_t& back() const { return _values[_size - 1]; }

private:
    std::size_t _values[max_size];
    std::size_t _size;
};

inline bool operator==(const Sizes& a, const Sizes& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}
inline bool operator!=(const Sizes& a, const Sizes& b) { return !(a == b); }

template<class T, int _dim>
class TensorView {
public:
    using DType = T;
    static const int dim = _dim;
    static const bool is_single = false;

    TensorView(uint8_t* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    TensorView(T& value) :
        _data(reinterpret_cast<uint8_t*>(&value)),
        _stride(nullptr),
        _shape(nullptr) {}

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    const TensorView<T, _dim-1> operator[](std::size_t index) const {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    TensorView<T, _dim-1> operator[](std::size_t index) {
        return {_data + index * _stride[0], _stride + 1, _shape + 1};
    }

    operator typename std::conditional_t<_dim == 0, T, Nothing>() const {
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
    virtual ~Device() = default;
    virtual void* allocate(std::size_t size) const = 0;
    virtual void free(void* ptr) const = 0;
    virtual void memcpy(void* to, void* from, std::size_t size) const = 0;
};

using DevicePtr = std::shared_ptr<Device>;

class CpuDevice : public Device {
public:
    void* allocate(std::size_t size) const override {
        return new uint8_t[size];
    }

    void free(void* ptr) const override {
        delete[] static_cast<uint8_t*>(ptr);
    }

    void memcpy(void* to, void* from, std::size_t size) const override {
        auto to_u8 = static_cast<uint8_t*>(to);
        auto from_u8 = static_cast<uint8_t*>(from);
        std::copy(from_u8, from_u8 + size, to_u8);
    }

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    friend inline DevicePtr cpu_device();

private:
    CpuDevice() {}
};

inline DevicePtr cpu_device() {
    static DevicePtr device = DevicePtr(new CpuDevice());
    return device;
}

class Tensor {
public:
    Tensor() : impl(nullptr) {}

    Tensor(const Tensor& other) : impl(other.impl) {
        if (impl != nullptr) ++impl->ref_count;
    }

    Tensor(Tensor&& other) noexcept : impl(other.impl) {
        other.impl = nullptr;
    }

    Tensor(DataType dtype, const Sizes& shape) :
        Tensor(dtype, shape, cpu_device()) {}

    Tensor(DataType dtype, const Sizes& shape, DevicePtr device) :
        impl(new TensorImpl{dtype, shape, device})
    {
        auto size = init_stride();
        impl->data = device->allocate(size);
    }

    Tensor(DataType dtype, const Sizes& shape, std::function<void*(std::size_t)> allocator) :
        Tensor(dtype, shape, cpu_device(), allocator) {}

    Tensor(
        DataType dtype,
        const Sizes& shape,
        DevicePtr device,
        std::function<void*(std::size_t)> allocator
    ) : impl(new TensorImpl{dtype, shape, device})
    {
        auto size = init_stride();
        impl->data = allocator(size);
    }

    Tensor(DataType dtype, const Sizes& shape, void* data) :
        Tensor(dtype, shape, cpu_device(), data) {}

    Tensor(DataType dtype, const Sizes& shape, DevicePtr device, void* data) :
        impl(new TensorImpl{dtype, shape, device, data, false})
    {
        init_stride();
    }

    Tensor(const SizeVec& batch_sizes) : impl(new TensorImpl{
        DataType::batch_sizes, {}, cpu_device(), nullptr, true, nullptr,
        1, {}, 0, batch_sizes
    }) {}

    template<typename T, typename = std::enable_if_t<
        std::is_same_v<T, bool> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>
    >>
    Tensor(T value, DevicePtr device) :
        impl(new TensorImpl{
            std::is_same_v<T, bool> ? DataType::dt_bool :
            std::is_same_v<T, int64_t> ? DataType::dt_int :
            DataType::dt_float,
            {1},
            device
        })
    {
        auto size = init_stride();
        impl->data = device->allocate(size);
        device->memcpy(impl->data, &value, sizeof(value));
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

    Tensor& operator=(Tensor&& other) noexcept {
        reset();
        impl = other.impl;
        other.impl = nullptr;
        return *this;
    }

    operator bool() {
        return impl != nullptr;
    }

    template<class T, int dim> TensorView<T, dim> view() {
        return TensorView<T, dim>(
            bytes(), impl->stride.data(), impl->shape.data()
        );
    }

    void* data() {
        return impl->data;
    }

    uint8_t* bytes() {
        return static_cast<uint8_t*>(impl->data) + impl->offset;
    }

    const Sizes& shape() const {
        return impl->shape;
    }

    const Sizes& stride() const {
        return impl->stride;
    }

    std::size_t size(std::size_t i) const {
        return impl->shape[i];
    }

    DataType dtype() const {
        return impl->dtype;
    }

    std::size_t dtype_size() const {
        switch (impl->dtype) {
            case DataType::dt_bool: return sizeof(bool);
            case DataType::dt_int: return sizeof(int64_t);
            case DataType::dt_float: return sizeof(double);
            case DataType::batch_sizes: return 0;
        }
    }

    std::size_t byte_size() const {
        std::size_t size = dtype_size();
        for (auto dim_size : impl->shape) {
            size *= dim_size;
        }
        return size;
    }

    const SizeVec& batch_sizes() const {
        return impl->batch_sizes;
    }

    void copy_from(void* source) {
        impl->device->memcpy(impl->data, source, impl->stride.back() * impl->shape.back());
    }

    void reset() {
        if (impl == nullptr) return;
        impl->reset();
        impl = nullptr;
    }

    void reset(std::function<void(void*)> deleter) {
        if (impl == nullptr) return;
        impl->reset(deleter);
        impl = nullptr;
    }

    Tensor select(std::size_t axis, std::size_t index);
    Tensor slice(std::size_t axis, std::size_t start, std::size_t stop);
    std::vector<Tensor> split(std::size_t axis, SizeVec sizes);
    std::vector<Tensor> unstack(std::size_t axis);
    Tensor cpu() { return *this; } //TODO: implement
    Tensor contiguous() { return *this; } //TODO: implement

private:
    struct TensorImpl {
        DataType dtype;
        Sizes shape;
        DevicePtr device;
        void* data;
        bool owns_data = true;
        TensorImpl* data_owner;
        int ref_count = 1;
        Sizes stride;
        std::size_t offset;
        SizeVec batch_sizes;

        void reset() {
            if (ref_count > 1) {
                --ref_count;
                return;
            }
            if (owns_data) {
                device->free(data);
            } else if (data_owner != nullptr) {
                data_owner->reset();
            }
            delete this;
        }

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

    Tensor(TensorImpl* _impl) : impl(_impl) {
        if (impl->data_owner != nullptr) {
            ++impl->data_owner->ref_count;
        }
    }
    std::size_t init_stride();
    TensorImpl* impl;
};


}
