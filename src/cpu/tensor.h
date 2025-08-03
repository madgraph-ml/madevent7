#pragma once

#include "madevent/runtime/tensor.h"
#include "device.h"
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

    template<typename... I>
    const VectorizedTensorView<V, T, _dim - sizeof...(I) - 1, false> get(
        std::size_t first_index, I... index
    ) const
    requires (_dim >= sizeof...(I) + 1)
    {
        T* ptr = _data;
        if constexpr (is_batch) {
            ptr = &ptr[first_index * _stride[0] * simd_vec_size];
        } else {
            ptr = &ptr[first_index * _stride[0]];
        }
        int i = 1;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I) + 1], &_shape[sizeof...(I) + 1], _batch_stride};
    }

    template<typename... I>
    VectorizedTensorView<V, T, _dim - sizeof...(I) - 1, false> get(
        std::size_t first_index, I... index
    )
    requires (_dim >= sizeof...(I) + 1)
    {
        T* ptr = _data;
        if constexpr (is_batch) {
            ptr = &ptr[first_index * _stride[0] * simd_vec_size];
        } else {
            ptr = &ptr[first_index * _stride[0]];
        }
        int i = 1;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I) + 1], &_shape[sizeof...(I) + 1], _batch_stride};
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

    std::size_t size(std::size_t index = 0) const {
        if (is_batch && index == 0) {
            return _shape[0] / simd_vec_size;
        } else {
            return _shape[index];
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

// return the tuple of flattened PackedTensorViews where the type is extracted
// from the signature of F
template<typename F, int dims> struct get_flat_views;
template<typename... TParam, int dims>
struct get_flat_views<void(*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(std::size_t flatten_count, TArg&&... args) {
        return std::make_tuple(
            args->template flat_view<typename TParam::DType, TParam::dim + dims>(
                flatten_count
            )...
        );
    }
};

// return tuple of TensorViews from PackedTensorViews
struct get_views {
    template <typename... TArg>
    auto operator()(TArg&... args) {
        return std::make_tuple(TensorView<typename TArg::DType, TArg::dim>(args)...);
    }
};

// return the tuple of VectorizedTensorViews where the type is extracted
// from the signature of F
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
inline void nested_for(std::size_t batch_size, std::size_t batch_offset, V... views) {
    auto& first_view = std::get<0>(std::tie(views...));
    if constexpr (dims == 0) {
        func(views...);
    } else if constexpr (dims == 1) {
        for (std::size_t i = batch_offset; i < batch_size; ++i) {
            func(views[i]...);
        }
    } else if constexpr (dims == 2) {
        auto size1 = first_view.size(1);
        for (std::size_t j = 0; j < size1; ++j) {
            for (std::size_t i = batch_offset; i < batch_size; ++i) {
                func(views.get(i, j)...);
            }
        }
    } else if constexpr (dims == 3) {
        auto size1 = first_view.size(1);
        auto size2 = first_view.size(2);
        for (std::size_t k = 0; k < size2; ++k) {
            for (std::size_t j = 0; j < size1; ++j) {
                for (std::size_t i = batch_offset; i < batch_size; ++i) {
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
                    for (std::size_t i = batch_offset; i < batch_size; ++i) {
                        func(views.get(i, j, k, l)...);
                    }
                }
            }
        }
    }
}

template<auto func, int dims, typename... V>
inline void nested_for_nobatch(V... views) {
    if constexpr (dims == 0) {
        func(views...);
    } else {
        auto& first_view = std::get<0>(std::tie(views...));
        nested_for<func, dims>(first_view.size(0), 0, views...);
    }
}

/*template<auto scalar_func, auto vector_func, int dims, typename V>
inline void tensor_foreach_body(std::size_t count, std::size_t offset, V& flat_views) {
    auto views = std::apply(
        get_views<decltype(scalar_func), dims>(), flat_views
    );
    std::size_t scalar_offset = offset;
    if constexpr (!std::is_same_v<decltype(scalar_func), decltype(vector_func)>) {
        auto vectorized_views = std::apply(
            get_vectorized_views<decltype(vector_func), dims>(), views
        );
        std::size_t vec_count = count / simd_vec_size;
        std::apply([vec_count, offset](auto&&... args) {
            nested_for<vector_func, dims>(vec_count, offset, args...);
        }, vectorized_views);
        scalar_offset += vec_count * simd_vec_size;
    }
    std::size_t scalar_count = offset + count - scalar_offset;
    std::apply([scalar_count, scalar_offset](auto&&... args) {
        nested_for<scalar_func, dims>(scalar_count, scalar_offset, args...);
    }, views);
}*/

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims, typename D>
inline void tensor_foreach_impl(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    std::size_t flatten_count,
    const D& device
) {
    // get views to the tensors with the correct types based on the signature of scalar_func
    auto flat_views = std::apply(
        get_flat_views<decltype(scalar_func), dims>(),
        std::tuple_cat(std::make_tuple(flatten_count), inputs, outputs)
    );

    device.foreach(
        batch_size,
        [flat_views] (std::size_t count, std::size_t offset, std::size_t thread_id) mutable {
            auto views = std::apply(
                get_views(), flat_views
            );
            std::size_t scalar_offset = offset;
            if constexpr (!std::is_same_v<decltype(scalar_func), decltype(vector_func)>) {
                auto vectorized_views = std::apply(
                    get_vectorized_views<decltype(vector_func), dims>(), views
                );
                std::size_t vec_count = count / simd_vec_size;
                std::apply([vec_count, offset](auto&&... args) {
                    nested_for<vector_func, dims>(vec_count, offset, args...);
                }, vectorized_views);
                scalar_offset += vec_count * simd_vec_size;
            }
            std::size_t scalar_count = offset + count - scalar_offset;
            std::apply([scalar_count, scalar_offset](auto&&... args) {
                nested_for<scalar_func, dims>(scalar_count, scalar_offset, args...);
            }, views);
        }
    );

    /*if constexpr (device.is_concurrent) {
        auto [job_count, job_size] = job_count_and_size(batch_size);
        int result = device.add_jobs(job_size);
        for (std::size_t i = 0; i < job_count; ++i) {
            std::size_t offset = i * job_size;
            std::size_t count = std::min(job_size, batch_size - offset);
            default_thread_pool().submit(
                [count, offset, flat_views, result](std::size_t thread_id) {
                    tensor_foreach_body<scalar_func, vector_func, dims>(
                        count, offset, flat_views
                    );
                    return result;
                }
            );
        }
    } else {
        tensor_foreach_body<scalar_func, vector_func, dims>(batch_size, 0, flat_views);
    }*/
}

template<auto scalar_func, auto vector_func, int n_in, int n_out, typename D>
inline void tensor_foreach_dynamic_impl(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    std::size_t iter_dims,
    const D& device
) {
    std::size_t flatten_count = iter_dims;
    for (auto input : inputs) {
        flatten_count = std::min(flatten_count, input->contiguous_dims());
    }
    for (auto output : outputs) {
        flatten_count = std::min(flatten_count, output->contiguous_dims());
    }
    if (flatten_count > 1) {
        iter_dims -= flatten_count - 1;
        auto& first_shape = std::get<0>(inputs)->shape();
        for (std::size_t i = 1; i < flatten_count; ++i) {
            batch_size *= first_shape[i];
        }
    }

    switch (iter_dims) {
        case 1:
            tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 1>(
                inputs, outputs, batch_size, flatten_count, device
            );
            break;
        case 2:
            tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 2>(
                inputs, outputs, batch_size, flatten_count, device
            );
            break;
        case 3:
            tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 3>(
                inputs, outputs, batch_size, flatten_count, device
            );
            break;
        case 4:
            tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 4>(
                inputs, outputs, batch_size, flatten_count, device
            );
            break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims, typename D>
inline void tensor_foreach(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    const D& device
) {
    if (batch_size == 0) return;
    if constexpr (dims > 1) {
        // call the dynamic foreach here as we can potentially be more efficient by
        // flattening contiguous dimensions
        tensor_foreach_dynamic_impl<scalar_func, vector_func, n_in, n_out>(
            inputs, outputs, batch_size, dims, device
        );
    } else {
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, dims>(
            inputs, outputs, batch_size, 0, device
        );
    }
}


template<auto scalar_func, auto vector_func, int n_in, int n_out, typename D>
inline void tensor_foreach_dynamic(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    const D& device
) {
    if (batch_size == 0) return;
    tensor_foreach_dynamic_impl<scalar_func, vector_func, n_in, n_out>(
        inputs,
        outputs,
        batch_size,
        std::get<0>(inputs)->shape().size() - first_param<decltype(scalar_func)>::dim,
        device
    );
}

}
}
