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
        bool differentiable;
    };

    Runtime(const Function& function) : Runtime(function, Context::default_context()) {}
    Runtime(const Function& function, ContextPtr context);
    TensorVec run(const TensorVec& inputs) const;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const;
    std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    );

private:
    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::size_t input_count;
    TensorVec locals_init;
    std::vector<bool> requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> grad_global_indices;
    ContextPtr context;
};

}
}
