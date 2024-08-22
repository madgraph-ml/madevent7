#pragma once

#include "madevent/backend/cuda/tensor.h"
#include "madevent/madcode/function.h"

namespace madevent {
namespace cuda {

class Runtime {
public:
    Runtime(const Function& function);
    std::vector<Tensor> run(std::vector<Tensor>& inputs) const;

private:
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        cudaStream_t stream;
        cudaEvent_t event;
    };

    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::vector<Tensor> locals_init;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
};

}
}
