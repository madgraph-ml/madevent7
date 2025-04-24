#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/madcode/function.h"

#include <memory>

namespace madevent {
namespace cuda {

class CudaRuntime : public Runtime {
public:
    struct CudaInstruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
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
    std::vector<CudaInstruction> instructions;
    SizeVec output_indices;
    std::vector<Tensor> locals_init;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
};

}
}
