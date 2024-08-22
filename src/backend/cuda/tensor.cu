#include "madevent/backend/cpu/tensor.h"

using namespace madevent::cuda;

void Tensor::reset() {
    if (impl.ref_count == nullptr) return;

    if (*impl.ref_count == 1) {
        delete impl.ref_count;
        cudaFree(impl.data);
    } else {
        --(*impl.ref_count);
    }
}

void Tensor::reset_async(cudaStream_t stream) {
    if (impl.ref_count == nullptr) return;

    if (*impl.ref_count == 1) {
        delete impl.ref_count;
        impl.ref_count = nullptr;
        cudaFreeAsync(impl.data, stream);
    } else {
        --(*ref_count);
    }
}

std::size_t Tensor::init_stride() {
    std::size_t stride_prod;
    switch (impl.dtype) {
        case DT_BOOL: stride_prod = sizeof(bool); break;
        case DT_INT: stride_prod = sizeof(int); break;
        case DT_FLOAT: stride_prod = sizeof(double); break;
    }
    bool first = true;
    std::size_t size_prod = 1;
    for (auto size : impl.shape) {
        if (first && size == 1) {
            stride.push_back(0);
        } else {
            stride.push_back(stride_prod);
        }
        if (first) {
            first = false;
        } else {
            size_prod *= size;
        }
        stride_prod *= size;
    }
    flat_shape = {impl.shape[0], size_prod};
    return stride_prod;
}
