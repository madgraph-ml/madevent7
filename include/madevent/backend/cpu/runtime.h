#pragma once

#include <vector>

#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using SizeVec = std::vector<std::size_t>;

template<class T, bool batch = false>
class TensorView {
public:
    using DType = T;

    TensorView(uint8_t* _data, std::size_t* _stride) : data(_data), stride(_stride) {}
    const TensorView<T> operator[](std::size_t index) const {
        // We always know the stride at compile time for the batch dimension
        // This should allow for some optimizations when loading data,
        // e.g. for SIMD vectorization
        //return TensorView<T>(data + index * (batch ? sizeof(T) : stride[0]), stride + 1);
        return TensorView<T>(data + index * stride[0], stride + 1);
    }
    TensorView<T> operator[](std::size_t index) {
        //return TensorView<T>(data + index * (batch ? sizeof(T) : stride[0]), stride + 1);
        return TensorView<T>(data + index * stride[0], stride + 1);
    }
    operator T() const { return *reinterpret_cast<T* const>(data); }
    T operator=(T value) { *reinterpret_cast<T*>(data) = value; return value; }

private:
    uint8_t* data;
    std::size_t* stride;
};


class Tensor {
public:
    using DataPtr = std::shared_ptr<uint8_t[]>;
    Tensor(DataType dtype, SizeVec shape, DataPtr data = nullptr);
    template<class T> TensorView<T, true> view() {
        return TensorView<T, true>(data.get(), stride.data());
    }
    std::size_t size(std::size_t i) { return shape[i]; }

//private:
    DataPtr data;
    DataType dtype;
    SizeVec shape;
    SizeVec stride;
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
