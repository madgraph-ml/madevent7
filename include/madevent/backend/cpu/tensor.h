#pragma once

// return the tuple of TensorViews where the type is extracted from the signature of F
template<typename F, int dims> struct get_views;
template<typename... TParam, int dims>
struct get_views<void(*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple(
            args.template view<typename TParam::DType, TParam::dim + dims>()...
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

template<auto scalar_func, auto vector_func, int n_in, int n_out, int dims>
void tensor_foreach(
    std::array<Tensor, n_in> inputs, std::array<Tensor, n_out> outputs, std::size_t batch_size
) {
    // get views to the tensors with the correct types based on the signature of scalar_func
    auto views = std::apply(
        get_views<decltype(scalar_func), dims>(), std::tuple_cat(inputs, outputs)
    );
    auto vectorized_views = std::apply(
        get_vectorized_views<decltype(vector_func), dims>(), views
    );
    std::size_t vec_batch_size = batch_size / simd_vec_size;
    for (std::size_t i = 0; i < vec_batch_size; ++i) {
        std::apply([i](auto&&... args) {
            recursive_for<vector_func, dims-1>(args[i]...);
        }, vectorized_views);
    }
    for (std::size_t i = vec_batch_size * simd_vec_size; i < batch_size; ++i) {
        std::apply([i](auto&&... args) {
            recursive_for<scalar_func, dims-1>(args[i]...);
        }, views);
    }
}

template<auto scalar_func, auto vector_func, int n_in, int n_out>
void tensor_foreach_dynamic(
    std::array<Tensor, n_in> inputs, std::array<Tensor, n_out> outputs, std::size_t batch_size
) {
    switch (std::get<0>(inputs).shape().size() - first_param<decltype(scalar_func)>::dim) {
        case 1:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 1>(inputs, outputs, batch_size);
            break;
        case 2:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 2>(inputs, outputs, batch_size);
            break;
        case 3:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 3>(inputs, outputs, batch_size);
            break;
        case 4:
            tensor_foreach<scalar_func, vector_func, n_in, n_out, 4>(inputs, outputs, batch_size);
            break;
        default:
            throw std::runtime_error("The number of tensor must be between 1 and 4");
    }
}

void tensor_copy(Tensor source, Tensor target) {
    tensor_foreach_dynamic<kernel_copy<CpuTypes>, kernel_copy<SimdTypes>, 1, 1>(
        {source}, {target}, target.size(0)
    );
}
