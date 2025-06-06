#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/thread_pool.h"
#include "simd.h"

namespace madevent {
namespace cpu {

template<class V, ScalarType T, int _dim, bool is_batch>
class VectorizedTensorView {
public:
    using VType = V;
    using DType = T;
    static const int dim = _dim;

    VectorizedTensorView(const TensorView<T, _dim>& view) :
        _data(view.data()), _stride(view.stride()), _shape(view.shape()),
        _batch_stride(view.stride()[0]) {}

    VectorizedTensorView(
        T* data, std::size_t* stride, std::size_t* shape, std::size_t batch_stride
    ) : _data(data), _stride(stride), _shape(shape), _batch_stride(batch_stride) {}

    VectorizedTensorView(V& value) :
        _data(reinterpret_cast<T*>(&value)),
        _stride(nullptr),
        _shape(nullptr),
        _batch_stride(0) {}

    const VectorizedTensorView<V, T, _dim-1, false> operator[](std::size_t index) const
    requires (_dim != 0)
    {
        if constexpr (is_batch) {
            return {
                &_data[index * _stride[0] * simd_vec_size],
                &_stride[1],
                &_shape[1],
                _batch_stride
            };
        } else {
            return {&_data[index * _stride[0]], &_stride[1], &_shape[1], _batch_stride};
        }
    }

    VectorizedTensorView<V, T, _dim-1, false> operator[](std::size_t index)
    requires (_dim != 0)
    {
        if constexpr (is_batch) {
            return {
                &_data[index * _stride[0] * simd_vec_size],
                &_stride[1],
                &_shape[1],
                _batch_stride
            };
        } else {
            return {&_data[index * _stride[0]], &_stride[1], &_shape[1], _batch_stride};
        }
    }

    operator V() const requires (_dim == 0) {
        return vload(_data, _batch_stride);
    }

    V operator=(V value) requires (_dim == 0) {
        vstore(_data, _batch_stride, value);
        return value;
    }

    V operator+=(V value) requires (_dim == 0) {
        V new_value = vload(_data, _batch_stride) + value;
        vstore(_data, _batch_stride, new_value);
        return new_value;
    }

    VectorizedTensorView<V, T, _dim, is_batch>& operator=(
        VectorizedTensorView<V, T, _dim, is_batch>& value
    ) = delete;

    std::size_t size() const {
        if constexpr (is_batch) {
            return _shape[0] / simd_vec_size;
        } else {
            return _shape[0];
        }
    }

    template<typename IVec>
    V gather(IVec indices) const requires (_dim == 1) {
        return vgather(_data, _batch_stride, _stride[0], indices);
    }

    template<typename IVec>
    void scatter_add(IVec indices, V values) requires (_dim == 1) {
        V old_values = vgather(_data, _batch_stride, _stride[0], indices);
        vscatter(_data, _batch_stride, _stride[0], indices, old_values + values);
    }

private:
    T* _data;
    std::size_t* _stride;
    std::size_t* _shape;
    std::size_t _batch_stride;
};

// return the tuple of TensorViews where the type is extracted from the signature of F
template<typename F, int dims> struct get_views;
template<typename... TParam, int dims>
struct get_views<void(*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple(
            args->template view<typename TParam::DType, TParam::dim + dims>()...
        );
    }
};

// return the tuple of VectorizedTensorViews where the type is extracted from the signature of F
template<typename F, int dims> struct get_vectorized_views;
template<typename... TParam, int dims>
struct get_vectorized_views<void(*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple(VectorizedTensorView<
            typename TParam::VType, typename TParam::DType, TParam::dim + dims, true
        >(args)...);
    }
};

template<typename F> struct first_param;
template<typename... TParam>
struct first_param<void(*)(TParam...)> {
    static constexpr int dim = std::get<0>(std::tie(TParam::dim...));
};

template<auto func, int dims, typename... V>
void recursive_for(V... views) {
    if constexpr (dims == 0) {
        func(views...);
    } else {
        std::size_t count = std::get<0>(std::tie(views...)).size();
        for (std::size_t i = 0; i < count; ++i) {
            recursive_for<func, dims-1>(views[i]...);
        }
    }
}

template<auto func, int dims, typename... V>
void nested_for(std::size_t batch_size, V... views) {
    auto& first_view = std::get<0>(std::tie(views...));
    if constexpr (dims == 0) {
        func(views...);
    } else if constexpr (dims == 1) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            func(views[i]...);
        }
    } else if constexpr (dims == 2) {
        auto size1 = first_view.size(1);
        for (std::size_t j = 0; j < size1; ++j) {
            for (std::size_t i = 0; i < batch_size; ++i) {
                func(views.get(i, j)...);
            }
        }
    } else if constexpr (dims == 3) {
        auto size1 = first_view.size(1);
        auto size2 = first_view.size(2);
        for (std::size_t k = 0; k < size2; ++k) {
            for (std::size_t j = 0; j < size1; ++j) {
                for (std::size_t i = 0; i < batch_size; ++i) {
                    func(views.get(i, j, k)...);
                }
            }
        }
    } else if constexpr (dims == 4) {
        auto size1 = first_view.size(1);
        auto size2 = first_view.size(2);
        auto size3 = first_view.size(3);
        for (std::size_t l = 0; l < size3; ++l) {
            for (std::size_t k = 0; k < size2; ++k) {
                for (std::size_t j = 0; j < size1; ++j) {
                    for (std::size_t i = 0; i < batch_size; ++i) {
                        func(views.get(i, j, k, l)...);
                    }
                }
            }
        }
    }
}

class CpuDevice;

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void tensor_foreach(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    const CpuDevice& device
) {
    // get views to the tensors with the correct types based on the signature of scalar_func
    auto views = std::apply(
        get_views<decltype(scalar_func), dims>(), std::tuple_cat(inputs, outputs)
    );
    // scalar func and vector func have the same type if cpu vectorization is turned off
    if constexpr (std::is_same_v<decltype(scalar_func), decltype(vector_func)>) {
        ThreadPool::instance().parallel_for([&](std::size_t i) {
            std::apply([i](auto&&... args) {
                recursive_for<scalar_func, dims-1>(args[i]...);
            }, views);
        }, batch_size);
        /*std::apply([batch_size](auto&&... args) {
            nested_for<scalar_func, dims>(batch_size, args...);
        }, views);*/
    } else {
        auto vectorized_views = std::apply(
            get_vectorized_views<decltype(vector_func), dims>(), views
        );
        std::size_t vec_batch_size = batch_size / simd_vec_size;
        ThreadPool::instance().parallel_for([&](std::size_t i) {
            std::apply([i](auto&&... args) {
                recursive_for<vector_func, dims-1>(args[i]...);
            }, vectorized_views);
        }, vec_batch_size);
        //for (std::size_t i = 0; i < vec_batch_size; ++i) {
        //    std::apply([i](auto&&... args) {
        //        recursive_for<vector_func, dims-1>(args[i]...);
        //    }, vectorized_views);
        //}
        for (std::size_t i = vec_batch_size * simd_vec_size; i < batch_size; ++i) {
            std::apply([i](auto&&... args) {
                recursive_for<scalar_func, dims-1>(args[i]...);
            }, views);
        }
    }
    /*for (std::size_t i = 0; i < batch_size; ++i) {
        std::apply([i](auto&&... args) {
            recursive_for<scalar_func, dims-1>(args[i]...);
        }, views);
    }*/
}

template<auto scalar_func, auto vector_func, int n_in, int n_out>
void tensor_foreach_dynamic(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    const CpuDevice& device
) {
    switch (std::get<0>(inputs)->shape().size() - first_param<decltype(scalar_func)>::dim) {
        case 1:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 1>(
                inputs, outputs, batch_size, device
            );
            break;
        case 2:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 2>(
                inputs, outputs, batch_size, device
            );
            break;
        case 3:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 3>(
                inputs, outputs, batch_size, device
            );
            break;
        case 4:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 4>(
                inputs, outputs, batch_size, device
            );
            break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

}
}
