#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/runtime_base.h"
#include "madevent/madcode/function.h"

#include <memory>
#include <cublas_v2.h>
#include <curand.h>

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
        CudaRuntime& runtime;
        bool differentiable;
        int stream;
        int backward_stream;
        std::vector<int> wait_events;
        int record_event;
        std::vector<int> backward_wait_events;
        int backward_record_event;
    };

    CudaRuntime(const Function& function, ContextPtr context);
    ~CudaRuntime();
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
    Context& context() { return *_context; }
    cublasHandle_t cublas_handle() { return _cublas_handle; }
    curandGenerator_t curand_generator() { return _curand_generator; }

private:
    std::vector<Instruction> _instructions;
    SizeVec _output_indices;
    std::size_t _input_count;
    TensorVec _locals_init;
    std::vector<bool> _requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> _grad_global_indices;
    ContextPtr _context;
    ThreadResource<std::vector<cudaStream_t>> _streams;
    ThreadResource<std::vector<cudaEvent_t>> _events;
    cublasHandle_t _cublas_handle;
    curandGenerator_t _curand_generator;
    int _sync_stream;
    std::vector<int> _sync_events;
    int _backward_sync_stream;
    std::vector<int> _backward_sync_events;
};

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context, bool concurrent);

}
}
