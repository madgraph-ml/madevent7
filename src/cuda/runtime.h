#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/runtime_base.h"
#include "madevent/madcode/function.h"

#include <memory>

namespace madevent {
namespace cuda {

class CudaRuntime : public Runtime {
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
        cudaStream_t stream;
        cudaEvent_t event;
    };

    CudaRuntime(const Function& function, ContextPtr context);
    TensorVec run(const TensorVec& inputs) const override;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const override;
    std::tuple<
        TensorVec, std::vector<std::tuple<std::string, Tensor>>
    > run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const override;

private:
    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::size_t input_count;
    TensorVec locals_init;
    std::vector<bool> requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> grad_global_indices;
    ContextPtr context;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
};

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context);

}
}
