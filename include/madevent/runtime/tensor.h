#pragma once

#include "madevent/madcode/type.h"
#include "madevent/util.h"

#include <cstdint>
#include <vector>
#include <functional>
#include <initializer_list>
#include <algorithm>
#include <concepts>
#include <atomic>

namespace madevent {

using SizeVec = std::vector<std::size_t>;

class Sizes {
public:
    static constexpr std::size_t max_size = 4;

    Sizes() : _size(0) {};
    Sizes(std::size_t size) : _size(size) {
        std::fill(begin(), end(), 0);
    };
    Sizes(std::size_t size, std::size_t value) : _size(size) {
        std::fill(begin(), end(), value);
    };
    Sizes(std::initializer_list<std::size_t> values) : _size(values.size()) {
        if (values.size() > max_size) {
            throw std::invalid_argument("maximum dimension exceeded");
        }
        std::copy(values.begin(), values.end(), begin());
    }
    Sizes(const SizeVec& values) : _size(values.size()) {
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

template<ScalarType T, int _dim>
struct PackedTensorView {
    using DType = T;
    static const int dim = _dim;
    T* data;
    Sizes stride;
    Sizes shape;
};

template<ScalarType T, int _dim>
class TensorView {
public:
    using DType = T;
    static const int dim = _dim;

    TensorView(T* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    TensorView(PackedTensorView<T, _dim>& packed_view) :
        _data(packed_view.data),
        _stride(packed_view.stride.data()),
        _shape(packed_view.shape.data())
    {}

    TensorView(T& value) : _data(&value), _stride(nullptr), _shape(nullptr) {}

    const TensorView<T, _dim-1> operator[](std::size_t index) const requires (_dim != 0) {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    TensorView<T, _dim-1> operator[](std::size_t index) requires (_dim != 0) {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    template<typename... I>
    const TensorView<T, _dim - sizeof...(I)> get(I... index) const
    requires (_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    template<typename... I>
    TensorView<T, _dim - sizeof...(I)> get(I... index)
    requires (_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    operator T() const requires (_dim == 0) {
        return *_data;
    }

    T operator=(T value) requires (_dim == 0) {
        *_data = value;
        return value;
    }

    T operator+=(T value) requires (_dim == 0) {
        *_data += value;
        return value;
    }

    TensorView<T, _dim>& operator=(TensorView<T, _dim>& value) = delete;
    std::size_t size(std::size_t index = 0) const { return _shape[index]; }
    T* data() const { return _data; }
    std::size_t* stride() const { return _stride; }
    std::size_t* shape() const { return _shape; }
    T gather(int64_t index) const requires (_dim == 1) { return (*this)[index]; }
    void scatter_add(int64_t index, T value) requires (_dim == 1) {
        (*this)[index] += value;
    }

private:
    T* _data;
    std::size_t* _stride;
    std::size_t* _shape;
};

class Tensor;

class Device {
public:
    virtual ~Device() = default;
    virtual void* allocate(std::size_t size) const = 0;
    virtual void free(void* ptr) const = 0;
    virtual void memcpy(void* to, void* from, std::size_t size) const = 0;
    virtual void tensor_copy(const Tensor& source, Tensor& target) const = 0;
    virtual void tensor_zero(Tensor& tensor) const = 0;
    virtual void tensor_add(const Tensor& source, Tensor& target) const = 0;
    virtual void tensor_cpu(const Tensor& source, Tensor& target) const = 0;
    virtual const Device* device_ptr() const = 0;
};

using DevicePtr = const Device*;
// defined in runtime_base.cpp, but need to declare them here
DevicePtr cpu_device();
DevicePtr cuda_device();

class Tensor {
public:
    Tensor() : impl(nullptr) {}

    Tensor(const Tensor& other) : impl(other.impl) {
        if (impl != nullptr) impl->incref();
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

    template<typename D>
    Tensor(DataType dtype, const Sizes& shape, const D& device) :
        impl(new TensorImpl{dtype, shape, device.device_ptr()})
    {
        auto size = init_stride();
        impl->data = device.allocate(size);
    }

    Tensor(
        DataType dtype,
        const Sizes& shape,
        void* data,
        std::function<void()> external_reset
    ) :
        Tensor(dtype, shape, cpu_device(), data, external_reset) {}

    Tensor(
        DataType dtype,
        const Sizes& shape,
        DevicePtr device,
        void* data,
        std::function<void()> external_reset
    ) :
        impl(new TensorImpl{dtype, shape, device, data, false, external_reset})
    {
        init_stride();
    }

    Tensor(
        DataType dtype,
        const Sizes& shape,
        const Sizes& stride,
        DevicePtr device,
        void* data,
        std::function<void()> external_reset
    ) :
        impl(new TensorImpl{
            dtype, shape, device, data, false, external_reset, nullptr, 1, stride
        })
    {
        std::size_t stride_prod = 1;
        bool first = true;
        impl->contiguous_dims = 0;
        for (auto [size_i, stride_i] : zip(shape, stride)) {
            if (stride_i == stride_prod) {
                ++impl->contiguous_dims;
            }
            if (first && size_i == 1) {
                impl->stride[0] = 0;
            }
            stride_prod *= size_i;
            first = false;
        }
    }

    Tensor(const SizeVec& batch_sizes) : impl(new TensorImpl{
        DataType::batch_sizes, {}, cpu_device(), nullptr, true, std::nullopt, nullptr,
        1, {}, 0, 0, batch_sizes
    }) {}

    template<ScalarType T>
    Tensor(T value, DevicePtr device) :
        impl(new TensorImpl{
            std::is_same_v<T, int64_t> ? DataType::dt_int : DataType::dt_float,
            {1},
            device
        })
    {
        auto size = init_stride();
        impl->data = device->allocate(size);
        device->memcpy(impl->data, &value, sizeof(value));
    }

    Tensor(TensorValue value, DevicePtr device) :
        impl(new TensorImpl{
            std::visit(Overloaded{
                [](std::vector<int64_t>) { return DataType::dt_int; },
                [](std::vector<double>) { return DataType::dt_float; },
            }, std::get<1>(value)),
            [&]{
                auto& val_shape = std::get<0>(value);
                Sizes full_shape(val_shape.size() + 1);
                full_shape[0] = 1;
                std::copy(val_shape.begin(), val_shape.end(), full_shape.begin() + 1);
                return full_shape;
            }(),
            device
        })
    {
        auto size = init_stride();
        impl->data = device->allocate(size);
        std::visit([&](auto& vec) {
            device->memcpy(impl->data, vec.data(), size);
        }, std::get<1>(value));
    }

    ~Tensor() {
        reset();
    }

    Tensor& operator=(const Tensor& other) {
        reset();
        impl = other.impl;
        if (impl != nullptr) impl->incref();
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        reset();
        impl = other.impl;
        other.impl = nullptr;
        return *this;
    }

    operator bool() const {
        return impl != nullptr;
    }

    template<class T, int dim>
    TensorView<T, dim> view() {
        check_impl();
        T* data = static_cast<T*>(impl->data);
        return TensorView<T, dim>(
            &data[impl->offset], impl->stride.data(), impl->shape.data()
        );
    }

    template<class T, int dim>
    const TensorView<T, dim> view() const {
        check_impl();
        T* data = static_cast<T*>(impl->data);
        return TensorView<T, dim>(
            &data[impl->offset], impl->stride.data(), impl->shape.data()
        );
    }

    template<class T, int dim>
    PackedTensorView<T, dim> flat_view(std::size_t flatten_count) const {
        check_impl();
        if (flatten_count <= 1) {
            return {static_cast<T*>(impl->data), impl->stride, impl->shape};
        }
        if (flatten_count > impl->contiguous_dims) {
            throw std::invalid_argument("can only flatten contiguous dimensions");
        }
        if (impl->stride[0] == 0) {
            throw std::runtime_error("ALARM!!!!!!!!");
        }
        Sizes stride{1}, shape{1};
        std::size_t i = 1;
        for (; i < flatten_count; ++i) {
            shape[0] *= impl->shape[i];
        }
        for (; i < impl->shape.size(); ++i) {
            shape.push_back(impl->shape[i]);
            stride.push_back(impl->stride[i]);
        }
        return {static_cast<T*>(impl->data), stride, shape};
    }

    void* data() { check_impl(); return impl->data; }
    void* data() const { check_impl(); return impl->data; }
    const Sizes& shape() const { check_impl(); return impl->shape; }
    const Sizes& stride() const { check_impl(); return impl->stride; }
    std::size_t size(std::size_t i) const { check_impl(); return impl->shape[i]; }
    std::size_t offset() const { return impl->offset; }
    DataType dtype() const { check_impl(); return impl->dtype; }
    const SizeVec& batch_sizes() const { check_impl(); return impl->batch_sizes; }
    DevicePtr device() const { check_impl(); return impl->device; }

    std::size_t dtype_size() const {
        check_impl();
        switch (impl->dtype) {
            case DataType::dt_int: return sizeof(int64_t);
            case DataType::dt_float: return sizeof(double);
            case DataType::batch_sizes: return 0;
            default: throw std::logic_error("invalid data type");
        }
    }

    std::size_t byte_size() const {
        check_impl();
        std::size_t size = dtype_size();
        for (auto dim_size : impl->shape) {
            size *= dim_size;
        }
        return size;
    }

    void reset() {
        if (impl == nullptr) return;
        impl->reset(*impl->device);
        impl = nullptr;
    }

    template<typename D>
    void reset(const D& device) {
        if (impl == nullptr) return;
        impl->reset(device);
        impl = nullptr;
    }

    Tensor select(std::size_t axis, std::size_t index) const;
    Tensor slice(std::size_t axis, std::size_t start, std::size_t stop) const;
    std::vector<Tensor> split(std::size_t axis, const SizeVec& sizes) const;
    std::vector<Tensor> unstack(std::size_t axis) const;
    Tensor unsqueeze(std::size_t axis) const;
    Tensor expand(const Sizes& shape) const;

    template<typename D>
    Tensor cpu(const D& device) const {
        check_impl();
        if (impl->device == cpu_device()) {
            return *this;
        } else {
            Tensor tensor(impl->dtype, impl->shape, impl->device);
            device.tensor_cpu(contiguous(device), tensor);
            return tensor;
        }
    }
    Tensor cpu() const { return cpu(*impl->device); }

    template<typename D>
    void zero(const D& device) {
        check_impl();
        device.tensor_zero(*this);
    }
    void zero() { zero(*impl->device); }

    template<typename D>
    void copy_from(const Tensor& source, const D& device) {
        check_impl();
        if (source.device() == this->device()) {
            device.tensor_copy(source, *this);
        } else if (is_contiguous()) {
            auto contig_source = source.contiguous();
            device.memcpy(data(), contig_source.data(), byte_size());
        } else {
            throw std::runtime_error("tensor must be contiguous for copy across devices");
        }
    }
    void copy_from(const Tensor& source) { copy_from(source, *impl->device); }

    template<typename D>
    void add(const Tensor& source, const D& device) {
        check_impl();
        device.tensor_add(source, *this);
    }
    void add(const Tensor& source) { add(source, *impl->device); }

    template<typename D>
    Tensor copy(const D& device) const {
        check_impl();
        Tensor tensor(impl->dtype, impl->shape, impl->device);
        device.tensor_copy(*this, tensor);
        return tensor;
    }
    Tensor copy() const { return copy(*impl->device); }

    bool is_contiguous() const {
        return impl->contiguous_dims == impl->shape.size();
    }

    std::size_t contiguous_dims() const {
        return impl->contiguous_dims > 0;
    }

    template<typename D>
    Tensor contiguous(const D& device) const {
        check_impl();
        return is_contiguous() ? *this : copy(device);
    }

    Tensor contiguous() const { return contiguous(*impl->device); }

    template<typename D>
    Tensor contiguous(std::size_t batch_size, const D& device) const {
        check_impl();
        if (size(0) == batch_size) {
            return contiguous(device);
        } else if (size(0) == 1) {
            auto shape = impl->shape;
            shape[0] = batch_size;
            Tensor tensor(impl->dtype, shape, impl->device);
            device.tensor_copy(*this, tensor);
            return tensor;
        } else {
            throw std::runtime_error("invalid batch size");
        }
    }

    Tensor contiguous(std::size_t batch_size) const {
        return contiguous(batch_size, *impl->device);
    }

private:
    struct TensorImpl {
        DataType dtype;
        Sizes shape;
        DevicePtr device;
        void* data;
        bool owns_data = true;
        std::optional<std::function<void()>> external_reset = std::nullopt;
        TensorImpl* data_owner;
        std::atomic<int> ref_count = 1;
        Sizes stride;
        std::size_t offset;
        std::size_t contiguous_dims;
        SizeVec batch_sizes;

        template<typename D>
        void reset(const D& device) {
            if (ref_count.fetch_sub(1, std::memory_order_acq_rel) != 1) {
                return;
            }
            if (owns_data) {
                device.free(data);
            } else if (data_owner != nullptr) {
                data_owner->reset(device);
            } else if (external_reset) {
                (*external_reset)();
            }
            delete this;
        }

        void incref() {
            ref_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    Tensor(TensorImpl* _impl) : impl(_impl) {
        if (impl->data_owner != nullptr) {
            impl->data_owner->incref();
        }
    }
    std::size_t init_stride();

    void check_impl() const {
        if (impl == nullptr) throw std::runtime_error("empty tensor");
    }

    TensorImpl* impl;
};

using TensorVec = std::vector<Tensor>;

}
