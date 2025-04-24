#include "runtime.h"

#include <tuple>
#include <array>
#include <functional>
#include <algorithm>

#include "madevent/cuda/tensor.h"
#include "madevent/cuda/device.h"
#include "madevent/util.h"
#include "../kernels/kernels.h"

using namespace madevent;
using namespace madevent_cuda;

namespace {

template<auto function, int NIn, int NOut, bool flatten>
void batch_foreach(CudaInstruction& instruction, std::vector<Tensor>& locals) {
    std::size_t batch_size = 1;
    auto inputs = range_to_tuple<NIn>([&](auto i) {
        auto& input = locals[instruction.input_indices[i]];
        auto input_size = input.size(0);
        if (input_size != 1) {
            if (batch_size == 1) {
                batch_size = input_size;
            } else if (input_size != batch_size) {
                throw std::runtime_error("incompatible input shapes");
            }
        }
        return input;
    });
    auto outputs = range_to_tuple<NOut>([&](auto i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        SizeVec shape {batch_size};
        shape.insert(shape.end(), output_shape.begin(), output_shape.end());
        output = Tensor(
            instruction.output_dtypes[i],
            shape,
            cuda_device(),
            [stream=instruction.stream](auto size) {
                void* ptr;
                cudaMallocAsync(&ptr, size, stream);
                return ptr;
            }
        );
        return output;
    });

    // get views to the tensors with the correct types based on the signature of function
    auto views = std::apply(
        get_views<decltype(function), flatten>(),
        std::tuple_cat(inputs, outputs)
    );

    int n_threads = 512;
    int n_blocks = (batch_size + n_threads - 1) / n_threads;
    std::apply([&](auto&&... args) {
        run_kernel<function><<<n_blocks, n_threads, 0, instruction.stream>>>
            (batch_size, args...);
    }, views);
}

}

Runtime::Runtime(const Function& function) : impl(std::make_unique<Impl>()) {
    impl->locals_init.resize(function.locals.size());
    InstructionDependencies deps(function);
    LastUseOfLocals last_use(function);
    std::vector<std::size_t> stream_last_instr;

    std::size_t instr_index = 0;
    for (auto& instr : function.instructions) {
        SizeVec input_indices;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
        }
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
        }
        impl->instructions.push_back({
            instr.instruction->opcode,
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes
        });
        for (auto local_index : last_use.locals(instr_index)) {
            impl->instructions.push_back({-1, {local_index}, {}, {}, {}});
        }
        ++instr_index;
    }

    for (auto& local : function.locals) {
        std::visit(Overloaded{
            [local, this](auto val) {
                Tensor tensor(local.type.dtype, {1}, cuda_device());
                cudaMemcpy(tensor.data(), &val, sizeof val, cudaMemcpyDefault);
                impl->locals_init[local.local_index] = tensor;
            },
            [](std::string val){},
            [](std::monostate val){}
        }, local.literal_value);
    }

    for (auto& out : function.outputs) {
        impl->output_indices.push_back(out.local_index);
    }
}

std::vector<Tensor> Runtime::run(std::vector<Tensor>& inputs) const {
    auto locals = impl->locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : impl->instructions) {
        switch (instr.opcode) {
            case -3: // record event
                cudaEventRecord(instr.event, instr.stream);
                break;
            case -2: // wait for event
                cudaStreamWaitEvent(instr.stream, instr.event);
                break;
            case -1: // free memory
                locals[instr.input_indices[0]].reset([stream = instr.stream] (void* ptr) {
                    cudaFreeAsync(ptr, stream);
                });
                break;
#include "runtime_mixin.h"
        }
    }
    cudaDeviceSynchronize();

    std::vector<Tensor> outputs;
    for (auto index : impl->output_indices) {
        outputs.push_back(locals[index]);
    }
    return outputs;
}
