#pragma once

#include "madevent/backend/tensor.h"
#include "madevent/backend/context.h"
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
        std::size_t batch_size_index;
        Context& context;
    };

    Runtime(const Function& function) {
        initialize(function, Context::default_context());
    }
    Runtime(const Function& function, Context& context) {
        initialize(function, context);
    }
    std::vector<madevent::Tensor> run(std::vector<madevent::Tensor>& inputs) const;

private:
    void initialize(const Function& function, Context& context);

    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::vector<madevent::Tensor> locals_init;
};

}
}
