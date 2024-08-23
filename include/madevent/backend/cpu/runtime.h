#pragma once

#include "madevent/backend/tensor.h"
#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

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
    std::vector<madevent::Tensor> run(std::vector<madevent::Tensor>& inputs) const;

private:
    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::vector<madevent::Tensor> locals_init;
};

}
}
