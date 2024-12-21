#include "madevent/backend/tensor.h"

using namespace madevent;

Tensor Tensor::select(std::size_t axis, std::size_t index) {
    auto new_shape = impl->shape;
    auto new_stride = impl->stride;
    new_shape.erase(new_shape.begin() + axis);
    new_stride.erase(new_stride.begin() + axis);
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        impl,
        1,
        new_stride,
        impl->offset + index * impl->stride[axis]
    });
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t stop) {
    auto new_shape = impl->shape;
    new_shape[axis] = stop - start;
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        impl,
        1,
        impl->stride,
        impl->offset + start * impl->stride[axis]
    });
}

std::vector<Tensor> Tensor::split(std::size_t axis, SizeVec sizes) {
    std::vector<Tensor> tensors;
    auto split_count = sizes.size();
    tensors.reserve(split_count);
    std::size_t offset = 0;
    for (std::size_t size : sizes) {
        auto next_offset = offset + size;
        tensors.push_back(slice(axis, offset, next_offset));
        offset = next_offset;
    }
    return tensors;
}

std::vector<Tensor> Tensor::unstack(std::size_t axis) {
    std::vector<Tensor> tensors;
    auto axis_size = size(axis);
    tensors.reserve(axis_size);
    for (std::size_t i = 0; i < axis_size; ++i) {
        tensors.push_back(select(axis, i));
    }
    return tensors;
}

std::size_t Tensor::contiguous_dims() {
    auto dt_size = dtype_size();
    std::size_t cont_dim = 0, size_prod = dtype_size();
    auto shape_iter = impl->shape.begin();
    for (auto stride : impl->stride) {
        if (stride != size_prod) break;
        size_prod *= *shape_iter;
        ++shape_iter;
        ++cont_dim;
    }
    return cont_dim;
}

std::size_t Tensor::init_stride() {
    std::size_t stride_prod = dtype_size();
    bool first = true;
    std::size_t size_prod = 1;
    for (auto size : impl->shape) {
        if (first && size == 1) {
            impl->stride.push_back(0);
        } else {
            impl->stride.push_back(stride_prod);
        }
        if (first) {
            first = false;
        } else {
            size_prod *= size;
        }
        stride_prod *= size;
    }
    return stride_prod;
}
