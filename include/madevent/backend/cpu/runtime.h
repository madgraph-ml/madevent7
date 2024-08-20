#pragma once

#include <vector>
#include <array>

#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using SizeVec = std::vector<std::size_t>;

template<class T, bool batch = false>
class TensorView {
public:
    using DType = T;

    TensorView(uint8_t* _data, std::size_t* _stride, std::size_t* _shape) :
        data(_data), stride(_stride), shape(_shape) {}
    const TensorView<T> operator[](std::size_t index) const {
        // We always know the stride at compile time for the batch dimension
        // This should allow for some optimizations when loading data,
        // e.g. for SIMD vectorization
        //return TensorView<T>(data + index * (batch ? sizeof(T) : stride[0]), stride + 1);
        return TensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    TensorView<T> operator[](std::size_t index) {
        //return TensorView<T>(data + index * (batch ? sizeof(T) : stride[0]), stride + 1);
        return TensorView<T>(data + index * stride[0], stride + 1, shape + 1);
    }
    operator T() const { return *reinterpret_cast<T* const>(data); }
    T operator=(T value) { *reinterpret_cast<T*>(data) = value; return value; }
    TensorView<T>& operator=(TensorView<T>& value) = delete;
    std::size_t size() const { return shape[0]; }

private:
    uint8_t* data;
    std::size_t* stride;
    std::size_t* shape;
};


class Tensor {
public:
    using DataPtr = std::shared_ptr<uint8_t[]>;
    Tensor(DataType dtype, SizeVec shape, DataPtr data = nullptr);
    template<class T> TensorView<T, true> view(bool flatten = false) {
        return TensorView<T, true>(
            data.get(), stride.data(), flatten ? flat_shape.data() : shape.data()
        );
    }
    std::size_t size(std::size_t i) { return shape[i]; }

//private:
    DataPtr data;
    DataType dtype;
    SizeVec shape;
    SizeVec stride;
    std::array<std::size_t, 2> flat_shape;
};


class Runtime {
public:
    using LocalVec = std::vector<std::optional<Tensor>>;
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
    };

    Runtime(const Function& function);
    std::vector<Tensor> run(std::vector<Tensor>& inputs) const;

private:
    std::vector<Instruction> instructions;
    SizeVec output_indices;
    LocalVec locals_init;
};

}
}
