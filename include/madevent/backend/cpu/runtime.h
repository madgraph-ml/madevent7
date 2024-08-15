#pragma once

#include <vector>

#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using SizeVec = std::vector<std::size_t>;

struct Untyped {};

template<class T, bool batch = false>
class TensorView {
public:
    TensorView(uint8_t* _data, std::size_t* _stride) : data(_data), stride(_stride) {}
    const TensorView<T> operator[](std::size_t index) const {
        //if (batch) {
            //return TensorView<T>(data + index, stride + 1);
        //} else {
            return TensorView<T>(data + index * stride[0], stride + 1);
        //}
    }
    TensorView<T> operator[](std::size_t index) {
        //if (batch) {
            //return TensorView<T>(data + index, stride + 1);
        //} else {
            return TensorView<T>(data + index * stride[0], stride + 1);
        //}
    }
    operator T() const { return *reinterpret_cast<T* const>(data); }
    //operator T&() { return *static_cast<T*>(data); }
    T operator=(T value) { *reinterpret_cast<T*>(data) = value; return value; }
    template <class T2> operator TensorView<T2>() { return TensorView<T2>(data, stride); }

private:
    uint8_t* data;
    std::size_t* stride;
};


class Tensor {
public:
    using DataPtr = std::shared_ptr<uint8_t[]>;
    Tensor(DataType dtype, SizeVec shape, DataPtr data = nullptr);
    //template<class T> operator TensorView<T>() { return TensorView<T>(data.get(), stride.data()); }
    TensorView<Untyped, true> view() { return TensorView<Untyped, true>(data.get(), stride.data()); }
    std::size_t size(std::size_t i) { return shape[i]; }

//private:
    DataPtr data;
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
    std::vector<Tensor> locals_init;
};

}
}
