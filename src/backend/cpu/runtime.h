#pragma once

#include <vector>

#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using Shape = std::vector<size_t>;

struct Tensor {
    uint8_t* data;
    DataType dtype;
    Shape shape;
}

class Runtime {
public:
    Runtime(const Function& function);
    std::vector<Tensor> run(std::vector<Tensor>& inputs) const;

private:
    std::vector<std::tuple<int, std::vector<size_t>, std::vector<size_t>> instructions;
    std::vector<std::tuple<size_t, DataType, Shape>> allocs;
}

}
}
