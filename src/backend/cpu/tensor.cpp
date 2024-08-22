#include "madevent/backend/cpu/tensor.h"

using namespace madevent::cpu;

Tensor::Tensor(DataType _dtype, SizeVec _shape, DataPtr _data)
    : dtype(_dtype), shape(_shape), data(_data)
{
    std::size_t stride_prod;
    switch (_dtype) {
        case DT_BOOL: stride_prod = sizeof(bool); break;
        case DT_INT: stride_prod = sizeof(int); break;
        case DT_FLOAT: stride_prod = sizeof(double); break;
    }
    bool first = true;
    std::size_t size_prod = 1;
    for (auto size : _shape) {
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
    flat_shape = {_shape[0], size_prod};
    if (!data) {
        data = std::shared_ptr<uint8_t[]>(new uint8_t[stride_prod]);
    }
}

