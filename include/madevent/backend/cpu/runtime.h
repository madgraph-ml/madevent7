#pragma once

#include <vector>

#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using SizeVec = std::vector<std::size_t>;

struct Untyped {};

template<class T>
class TensorView {
public:
    TensorView(uint8_t* _data, size_t* _stride) : data(_data), stride(_stride) {}
    const TensorView<T> operator[](size_t index) const {
        return TensorView(data + index * stride[0], stride + 1);
    }
    TensorView<T> operator[](size_t index) {
        return TensorView(data + index * stride[0], stride + 1);
    }
    operator T() const { return *reinterpret_cast<T* const>(data); }
    //operator T&() { return *static_cast<T*>(data); }
    T operator=(T value) { *reinterpret_cast<T*>(data) = value; return value; }
    template <class T2> operator TensorView<T2>() { return TensorView<T2>(data, stride); }

private:
    uint8_t* data;
    size_t* stride;
};


class Tensor {
public:
    Tensor(DataType dtype, SizeVec shape);
    template<class T> operator TensorView<T>() { return TensorView<T>(data->data(), stride.data()); }
    TensorView<Untyped> view() { return TensorView<Untyped>(data->data(), stride.data()); }
    size_t size(size_t i) { return shape[i]; }

private:
    std::shared_ptr<std::vector<uint8_t>> data;
    DataType dtype;
    SizeVec shape;
    SizeVec stride;
};


class Runtime {
public:
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
    size_t local_count;
};

}
}
