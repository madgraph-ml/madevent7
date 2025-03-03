#pragma once

#include "madevent/backend/tensor.h"
#include "madevent/backend/context.h"
#include "madevent/madcode/function.h"

namespace madevent {
namespace cpu {

using TensorVec = std::vector<madevent::Tensor>;

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
        bool eval_grad;
    };

    Runtime(const Function& function) {
        initialize(function, Context::default_context(), {});
    }
    Runtime(const Function& function, Context& context) {
        initialize(function, context, {});
    }
    Runtime(
        const Function& function,
        Context& context,
        const std::vector<std::string>& grad_globals
    ) {
        initialize(function, context, grad_globals);
    }
    TensorVec run(TensorVec& inputs) const;
    std::tuple<TensorVec, TensorVec> run_with_grad(TensorVec& inputs) const;
    std::unordered_map<std::string, Tensor> run_backward(
        TensorVec& output_grads, TensorVec& locals
    );

private:
    void initialize(
        const Function& function,
        Context& context,
        const std::vector<std::string>& grad_globals
    );

    std::vector<Instruction> instructions;
    SizeVec output_indices;
    TensorVec locals_init;
    std::unordered_map<std::string, std::size_t> grad_global_indices;
};

}
}
