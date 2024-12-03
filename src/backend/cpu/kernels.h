#pragma once

#include "madevent/backend/tensor.h"

#include <cmath>

#ifdef __aarch64__
#include "simd_arm.h"
#endif

namespace madevent {

template<class V, class T, int _dim, bool is_batch>
class VectorizedTensorView {
public:
    using VType = V;
    using DType = T;
    static const int dim = _dim;

    VectorizedTensorView(const TensorView<T, _dim>& view) :
        _data(view.data()), _stride(view.stride()), _shape(view.shape()),
        _batch_stride(view.stride()[0]) {}

    VectorizedTensorView(
        uint8_t* data, std::size_t* stride, std::size_t* shape, std::size_t batch_stride
    ) : _data(data), _stride(stride), _shape(shape), _batch_stride(batch_stride) {}

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    const VectorizedTensorView<V, T, _dim-1, false> operator[](std::size_t index) const {
        if (is_batch) {
            return {_data + index * _stride[0] * simd_vec_size, _stride + 1, _shape + 1, _batch_stride};
        } else {
            return {_data + index * _stride[0], _stride + 1, _shape + 1, _batch_stride};
        }
    }

    template<int d = _dim, typename = std::enable_if_t<d != 0>>
    VectorizedTensorView<V, T, _dim-1, false> operator[](std::size_t index) {
        if (is_batch) {
            return {_data + index * _stride[0] * simd_vec_size, _stride + 1, _shape + 1, _batch_stride};
        } else {
            return {_data + index * _stride[0], _stride + 1, _shape + 1, _batch_stride};
        }
    }

    operator typename std::conditional_t<_dim == 0, V, Nothing>() const {
        T buffer[simd_vec_size];
        for (int i = 0; i < simd_vec_size; ++i) {
            buffer[i] = *reinterpret_cast<T*>(_data + i * _batch_stride);
        }
        return vload(&buffer[0]);
    }

    template<int d = _dim, typename = std::enable_if_t<d == 0>>
    V operator=(V value) {
        T buffer[simd_vec_size];
        vstore(&buffer[0], value);
        for (int i = 0; i < simd_vec_size; ++i) {
            *reinterpret_cast<T*>(_data + i * _batch_stride) = buffer[i];
        }
        return value;
    }

    VectorizedTensorView<V, T, _dim, is_batch>& operator=(
        VectorizedTensorView<V, T, _dim, is_batch>& value
    ) = delete;

    std::size_t size() const {
        if (is_batch) {
            return _shape[0] / simd_vec_size;
        } else {
            return _shape[0];
        }
    }

private:
    uint8_t* _data;
    std::size_t* _stride;
    std::size_t* _shape;
    std::size_t _batch_stride;
};

}

#define KERNELSPEC

namespace {

struct CpuTypes {
    template<int dim> using FIn = const madevent::TensorView<double, dim>;
    template<int dim> using IIn = const madevent::TensorView<long long, dim>;
    template<int dim> using BIn = const madevent::TensorView<bool, dim>;
    template<int dim> using FOut = madevent::TensorView<double, dim>;
    template<int dim> using IOut = madevent::TensorView<long long, dim>;
    template<int dim> using BOut = madevent::TensorView<bool, dim>;
    using FVal = double;
    using IVal = long long;
    using BVal = bool;
};

template<typename T>
T where(bool condition, T val_true, T val_false) {
    return condition ? val_true : val_false;
}

using std::sqrt;
using std::sin;
using std::cos;
using std::sinh;
using std::cosh;
using std::atan2;
using std::pow;
using std::fabs;
using std::log;
using std::tan;
using std::atan;
using std::exp;

struct SimdTypes {
    template<int dim> using FIn = const madevent::VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IIn = const madevent::VectorizedTensorView<IVec, long long, dim, false>;
    template<int dim> using BIn = const madevent::VectorizedTensorView<BVec, bool, dim, false>;
    template<int dim> using FOut = madevent::VectorizedTensorView<FVec, double, dim, false>;
    template<int dim> using IOut = madevent::VectorizedTensorView<IVec, long long, dim, false>;
    template<int dim> using BOut = madevent::VectorizedTensorView<BVec, bool, dim, false>;
    using FVal = FVec;
    using IVal = IVec;
    using BVal = BVec;
};

#include "../common/kernels.h"

}
