#include "runtime.h"

#include <tuple>
#include <array>
#include <functional>
#include <algorithm>
#include <sstream>

#include "madevent/util.h"
#include "tensor.h"
#include "device.h"
#include "../kernels/kernels.h"
#include "../kernels/operations.h"

using namespace madevent;
using namespace madevent::cuda;
using namespace madevent::kernels;

namespace {

void op_matrix_element(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_matrix_element_multichannel(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_pdf(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_matmul(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void backward_op_matmul(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncCudaDevice& device
) {

}

void op_nonzero(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_batch_gather(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_scatter(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_random(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

void op_unweight(
    const CudaRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncCudaDevice& device
) {

}

}

CudaRuntime::CudaRuntime(const Function& function, ContextPtr context) :
    context(context), input_count(function.inputs().size())
{
    locals_init.resize(function.locals().size());
    requires_grad_init.resize(function.locals().size());
    auto opt_function = optimize_constants(function);
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(opt_function);

    for (auto& instr : opt_function.instructions()) {
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
        }
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
        }
        instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *context,
            instr.instruction->differentiable()
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back({
                -1, {local_index}, {}, {}, {}, 0, *context, false
            });
        }
        ++instr_index;
    }

    for (auto& [name, value] : opt_function.globals()) {
        Tensor global = context->global(name);
        auto& global_shape = value.type.shape;
        Sizes full_shape(global_shape.size() + 1);
        full_shape[0] = 1;
        std::copy(global_shape.begin(), global_shape.end(), full_shape.begin() + 1);
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            std::stringstream message;
            message << "Global " << name << " has wrong dtype or shape";
            throw std::invalid_argument(message.str());
        }
        locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            requires_grad_init.at(value.local_index) = true;
            grad_global_indices.push_back({name, value.local_index});
        }
    }

    for (auto& local : opt_function.locals()) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, &CudaDevice::instance());
                locals_init[local.local_index] = tensor;
            },
            [](std::monostate val){}
        }, local.literal_value);
    }

    for (auto& out : opt_function.outputs()) {
        output_indices.push_back(out.local_index);
    }
}

TensorVec CudaRuntime::run(const TensorVec& inputs) const {
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        AsyncCudaDevice device(instr.stream);
        switch (instr.opcode) {
            case -3: // record event
                cudaEventRecord(instr.event, instr.stream);
                break;
            case -2: // wait for event
                cudaStreamWaitEvent(instr.stream, instr.event);
                break;
            case -1: // free memory
                locals[instr.input_indices[0]].reset(device);
                break;
#include "runtime_mixin.h"
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    cudaDeviceSynchronize();
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CudaRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto locals = locals_init;
    auto requires_grad = requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto [instr, instr_eval_grad] : zip(instructions, eval_grad)) {
        AsyncCudaDevice device(instr.stream);
        if (instr.differentiable) {
            for (auto input_index : instr.input_indices) {
                if (requires_grad[input_index]) {
                    instr_eval_grad = true;
                    break;
                }
            }
            if (instr_eval_grad) {
                //TODO: only store necessary
                for (auto input_index : instr.input_indices) {
                    store_local[input_index] = true;
                }
                for (auto output_index : instr.output_indices) {
                    store_local[output_index] = true;
                    requires_grad[output_index] = true;
                }
            }
        }
        switch (instr.opcode) {
            case -3: // record event
                cudaEventRecord(instr.event, instr.stream);
                break;
            case -2: // wait for event
                cudaStreamWaitEvent(instr.stream, instr.event);
                break;
            case -1: { // free memory
                auto input_index = instr.input_indices[0];
                if (!store_local[input_index]) {
                    locals[input_index].reset(device);
                }
                break;
            }
#include "runtime_mixin.h"
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    cudaDeviceSynchronize();
    return {outputs, locals, eval_grad};
}

std::tuple<
    TensorVec, std::vector<std::tuple<std::string, Tensor>>
> CudaRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) const {
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (
        auto [instr, instr_eval_grad] :
        zip(std::views::reverse(instructions), std::views::reverse(eval_grad))
    ) {
        if (!instr_eval_grad) continue;
        bool needs_grad = true;
        for (auto [output_index, output_dtype] : zip(
            instr.output_indices, instr.output_dtypes
        )) {
            if (!local_grads[output_index] && output_dtype == DataType::dt_float) {
                needs_grad = false;
                break;
            }
        }
        if (needs_grad) {
            AsyncCudaDevice device(instr.stream);
            switch (instr.opcode) {
#include "runtime_backward_mixin.h"
            }
        }
    }
    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    cudaDeviceSynchronize();
    return {{local_grads.begin(), local_grads.begin() + input_count}, global_grads};
}

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context) {
    return new CudaRuntime(function, context);
}
