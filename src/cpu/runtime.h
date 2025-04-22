#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/runtime_base.h"
#include "madevent/runtime/context.h"
#include "madevent/madcode/function.h"

namespace madevent_cpu {

using TensorVec = std::vector<madevent::Tensor>;

class CpuRuntime : public madevent::Runtime {
public:
    struct Instruction {
        int opcode;
        madevent::SizeVec input_indices;
        madevent::SizeVec output_indices;
        std::vector<madevent::DataType> output_dtypes;
        std::vector<madevent::SizeVec> output_shapes;
        std::size_t batch_size_index;
        madevent::Context& context;
        bool differentiable;
    };

    CpuRuntime(const madevent::Function& function, madevent::ContextPtr context);
    TensorVec run(const TensorVec& inputs) const override;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const override;
    std::tuple<
        TensorVec, std::vector<std::tuple<std::string, madevent::Tensor>>
    > run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const override;

private:
    std::vector<Instruction> instructions;
    madevent::SizeVec output_indices;
    std::size_t input_count;
    TensorVec locals_init;
    std::vector<bool> requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> grad_global_indices;
    madevent::ContextPtr context;
};

extern "C" madevent::Runtime* build_runtime(
    const madevent::Function& function, madevent::ContextPtr context
);

}
