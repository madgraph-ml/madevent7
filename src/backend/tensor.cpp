#include "madevent/backend/tensor.h"

using namespace madevent;

std::size_t Tensor::init_stride() {
    std::size_t stride_prod;
    switch (impl->dtype) {
        case DT_BOOL: stride_prod = sizeof(bool); break;
        case DT_INT: stride_prod = sizeof(int); break;
        case DT_FLOAT: stride_prod = sizeof(double); break;
    }
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
    impl->flat_shape = {impl->shape[0], size_prod};
    return stride_prod;
}
