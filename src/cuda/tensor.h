#pragma once

#include "madevent/madcode/type.h"
#include "madevent/runtime/tensor.h"

#include <algorithm>

namespace madevent {
namespace cuda {

template<ScalarType T, int _dim, typename S>
class CudaTensorViewBase {
public:
    using DType = T;
    static const int dim = _dim;

    __device__ const TensorView<T, _dim-1> operator[](std::size_t index) const
    requires (_dim != 0)
    {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    __device__ TensorView<T, _dim-1> operator[](std::size_t index) requires (_dim != 0) {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    template<typename... I>
    __device__ const TensorView<T, _dim - sizeof...(I)> get(I... index) const
    requires (_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    template<typename... I>
    __device__ TensorView<T, _dim - sizeof...(I)> get(I... index)
    requires (_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    __device__ operator T() const requires (_dim == 0) {
        return *_data;
    }

    __device__ T operator=(T value) requires (_dim == 0) {
        *_data = value;
        return value;
    }

    __device__ T operator+=(T value) requires (_dim == 0) {
        *_data += value;
        return value;
    }

    __device__ TensorView<T, _dim>& operator=(TensorView<T, _dim>& value) = delete;
    __device__ T gather(int64_t index) const requires (_dim == 1) { return (*this)[index]; }
    __device__ void scatter_add(int64_t index, T value) requires (_dim == 1) {
        (*this)[index] += value;
    }

    __host__ __device__ std::size_t size(std::size_t index = 0) const {
        return _shape[index];
    }
    ___host__ _device__ T* data() const { return _data; }
    ___host__ _device__ std::size_t* stride() const { return _stride; }
    ___host__ _device__ std::size_t* shape() const { return _shape; }
private:
    T* _data;
    S _stride;
    S _shape;
};

template<ScalarType T, int _dim>
class CudaTensorView : public CudaTensorViewBase<T, _dim, std::size_t*> {
    __device__ CudaTensorView(T* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    __device__ CudaTensorView(T& value) : _data(&value), _stride(nullptr), _shape(nullptr) {}
};

template<class T, int _dim>
class PackedCudaTensorView : public CudaTensorViewBase<T, _dim, std::size_t[_dim]> {
public:
    PackedCudaTensorView(const TensorView<T, _dim>& view) :
        data(view.data())
    {
        if constexpr (_dim > 0) {
            std::copy(view.shape(), view.shape() + dim, std::begin(_shape));
            std::copy(view.stride(), view.stride() + dim, std::begin(_stride));
        }
    }
};

// return the tuple of PackedCudaTensorViews where the type is extracted from
// the signature of F
template<typename F, int dims> struct get_views;
template<typename... TParam, int dims>
struct get_views<void(*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple(
            PackedCudaTensorView(
                args->template view<typename TParam::DType, TParam::dim + dims>()
            )...
        );
    }
};

template<auto func, int dims, typename... V>
__global__ void run_kernel(std::size_t batch_size, V... views) {
    auto& first_view = std::get<0>(std::tie(views...));
    std::size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if constexpr (dims == 0) {
        func(views...);
    } else if constexpr (dims == 1) {
        if (index >= batch_size) return;
        func(views[index]...);
    } else if constexpr (dims == 2) {
        std::size_t size1 = first_view.size(1);
        if (index >= batch_size * size1) return;
        std::size_t i = index % batch_size;
        std::size_t j = index / batch_size;
        func(views.get(i, j)...);
    } else if constexpr (dims == 3) {
        std::size_t size1 = first_view.size(1);
        std::size_t size2 = first_view.size(2);
        if (index >= batch_size * size1 * size2) return;
        std::size_t i = index % batch_size;
        std::size_t jk = index / batch_size;
        std::size_t j = jk % size1;
        std::size_t k = jk / size1;
        func(views.get(i, j, k)...);
    } else if constexpr (dims == 4) {
        std::size_t size1 = first_view.size(1);
        std::size_t size2 = first_view.size(2);
        std::size_t size3 = first_view.size(3);
        if (index >= batch_size * size1 * size2) return;
        std::size_t i = index % batch_size;
        std::size_t jkl = index / batch_size;
        std::size_t j = jkl % size1;
        std::size_t kl = jkl / size1;
        std::size_t k = kl % size2;
        std::size_t l = kl / size2;
        func(views.get(i, j, k, l)...);
    }
}

template<auto func, int n_in, int n_out, int dims>
void tensor_foreach(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    cudaStream_t stream
) {
    // get views to the tensors with the correct types based on the signature of func
    auto views = std::apply(
        get_views<decltype(scalar_func), dims>(), std::tuple_cat(inputs, outputs)
    );
    auto& first_view = std::get<0>(std::tie(views...));

    std::size_t total_count = batch_size;
    for (std::size_t i = 1; i < dims; ++i) {
        total_count *= first_view.size(i);
    }

    std::size_t n_threads = 512;
    std::size_t n_blocks = (total_count + n_threads - 1) / n_threads;
    std::apply([&](auto&&... args) {
        run_kernel<function><<<n_blocks, n_threads, 0, stream>>>
            (batch_size, args...);
    }, views);
}

template<auto func, auto vector_func, int n_in, int n_out>
void tensor_foreach_dynamic(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    cudaStream_t stream
) {
    switch (std::get<0>(inputs)->shape().size() - first_param<decltype(scalar_func)>::dim) {
        case 1:
            tensor_foreach<func, n_in, n_out, 1>(inputs, outputs, batch_size, stream);
            break;
        case 2:
            tensor_foreach<func, n_in, n_out, 2>(inputs, outputs, batch_size, stream);
            break;
        case 3:
            tensor_foreach<func, n_in, n_out, 3>(inputs, outputs, batch_size, stream);
            break;
        case 4:
            tensor_foreach<func, n_in, n_out, 4>(inputs, outputs, batch_size, stream);
            break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

}
}
