#include "runtime.h"

#include <tuple>
#include <algorithm>
#include <ranges>

#include "madevent/madcode/optimizer.h"
#include "madevent/util.h"
#include "../kernels/kernels.h"
#include "../kernels/operations.h"
#include "device.h"


extern "C" void dgemm_(
    char* transa, char* transb,
    int* m, int* n, int* k,
    double* alpha,
    double* a, int* lda,
    double* b, int* ldb,
    double* beta,
    double* c, int* ldc
);

extern "C" void dgemv_(
    char* trans,
    int* m, int* n,
    double* alpha,
    double* a, int* lda,
    double* x, int* incx,
    double* beta,
    double* y, int* incy
);

using namespace madevent;
using namespace madevent::cpu;
using namespace madevent::kernels;

namespace {

void op_matrix_element(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size}, device);
    std::size_t me_index = locals[instruction.input_indices[3]].view<int64_t, 0>();
    auto& matrix_element = instruction.context.matrix_element(me_index);
    // TODO: maybe copy can be avoided sometimes
    auto momenta_in = locals[instruction.input_indices[0]].contiguous(batch_size, device);
    auto flavor_in = locals[instruction.input_indices[1]].contiguous(batch_size, device);
    auto mirror_in = locals[instruction.input_indices[2]].contiguous(batch_size, device);

    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != matrix_element.particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    auto& pool = ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto flavor_ptr = static_cast<int64_t*>(flavor_in.data());
    auto mirror_ptr = static_cast<int64_t*>(mirror_in.data());
    auto me_ptr = static_cast<double*>(me_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        matrix_element.call(
            matrix_element.process_instance(0), batch_size, batch_size,
            mom_ptr, flavor_ptr, mirror_ptr, me_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            matrix_element.call(
                matrix_element.process_instance(thread_id),
                std::min(batch_size - offset, count_per_thread),
                batch_size, mom_ptr + offset, flavor_ptr + offset,
                mirror_ptr + offset, me_ptr + offset
            );
        });
    }
}

void op_matrix_element_multichannel(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t me_index = locals[instruction.input_indices[5]].view<int64_t, 0>();
    std::size_t diagram_count = locals[instruction.input_indices[6]].view<int64_t, 0>();

    auto& me_out = locals[instruction.output_indices[0]];
    me_out = Tensor(DataType::dt_float, {batch_size}, device);
    auto& amp2_out = locals[instruction.output_indices[1]];
    amp2_out = Tensor(DataType::dt_float, {batch_size, diagram_count}, device);
    auto& diagram_out = locals[instruction.output_indices[2]];
    diagram_out = Tensor(DataType::dt_int, {batch_size}, device);
    auto& color_out = locals[instruction.output_indices[3]];
    color_out = Tensor(DataType::dt_int, {batch_size}, device);
    auto& helicity_out = locals[instruction.output_indices[4]];
    helicity_out = Tensor(DataType::dt_int, {batch_size}, device);

    auto& matrix_element = instruction.context.matrix_element(me_index);

    // TODO: maybe copy can be avoided sometimes
    auto momenta_in = locals[instruction.input_indices[0]].contiguous(batch_size, device);
    auto alpha_s_in = locals[instruction.input_indices[1]].contiguous(batch_size, device);
    auto random_in = locals[instruction.input_indices[2]].contiguous(batch_size, device);
    auto flavor_in = locals[instruction.input_indices[3]].contiguous(batch_size, device);
    auto mirror_in = locals[instruction.input_indices[4]].contiguous(batch_size, device);
    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != matrix_element.particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    if (diagram_count != matrix_element.diagram_count()) {
        throw std::runtime_error("Incompatible diagram count");
    }

    auto& pool = ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto alpha_ptr = static_cast<double*>(alpha_s_in.data());
    auto random_ptr = static_cast<double*>(random_in.data());
    auto flavor_ptr = static_cast<int64_t*>(flavor_in.data());
    auto mirror_ptr = static_cast<int64_t*>(mirror_in.data());
    auto me_ptr = static_cast<double*>(me_out.data());
    auto amp2_ptr = static_cast<double*>(amp2_out.data());
    auto diag_ptr = static_cast<int64_t*>(diagram_out.data());
    auto color_ptr = static_cast<int64_t*>(color_out.data());
    auto helicity_ptr = static_cast<int64_t*>(helicity_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        matrix_element.call_multichannel(
            matrix_element.process_instance(0), batch_size, batch_size,
            mom_ptr, alpha_ptr, random_ptr, flavor_ptr, mirror_ptr,
            me_ptr, amp2_ptr, diag_ptr, color_ptr, helicity_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            matrix_element.call_multichannel(
                matrix_element.process_instance(thread_id),
                std::min(batch_size - offset, count_per_thread), batch_size,
                mom_ptr + offset, alpha_ptr + offset, random_ptr + offset,
                flavor_ptr + offset, mirror_ptr + offset, me_ptr + offset,
                amp2_ptr + offset, color_ptr + offset, diag_ptr + offset,
                helicity_ptr + offset
            );
        });
    }
}

void op_matmul(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto bias = locals[instruction.input_indices[2]].contiguous(device);
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(DataType::dt_float, {batch_size, dims_out}, device);
    output.copy_from(bias, device);
    if (batch_size == 0) return;

    char transa = 'N', transb = 'T';
    int m = batch_size, n = dims_out, k = dims_in;
    double alpha = 1., beta = 1.;
    int lda = batch_size, ldb = dims_out, ldc = batch_size;
    double* a = static_cast<double*>(input.data());
    double* b = static_cast<double*>(weight.data());
    double* c = static_cast<double*>(output.data());
    dgemm_(
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
    );
}

void backward_op_matmul(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const CpuDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous(device);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, input.shape(), device);
        input_grad.zero();
    }
    if (!weight_grad) {
        weight_grad = Tensor(DataType::dt_float, weight.shape(), device);
        weight_grad.zero();
    }
    if (!bias_grad) {
        bias_grad = Tensor(DataType::dt_float, {1, dims_out}, device);
        bias_grad.zero();
    }
    if (batch_size == 0) return;

    // compute input_grad += output_grad * weight
    {
        char transa = 'N', transb = 'N';
        int m = batch_size, n = dims_in, k = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = dims_out, ldc = batch_size;
        double* a = static_cast<double*>(output_grad.data());
        double* b = static_cast<double*>(weight.data());
        double* c = static_cast<double*>(input_grad.data());
        dgemm_(
            &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
        );
    }

    // compute weight_grad += output_grad.T * input
    {
        char transa = 'T', transb = 'N';
        int m = dims_out, n = dims_in, k = batch_size;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = batch_size, ldc = dims_out;
        double* a = static_cast<double*>(output_grad.data());
        double* b = static_cast<double*>(input.data());
        double* c = static_cast<double*>(weight_grad.data());
        dgemm_(
            &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc
        );
    }

    // compute bias_grad += sum_i output_grad_ij
    {
        // TODO: we should probably do this differently...
        std::vector<double> ones(batch_size, 1.);
        char trans = 'T';
        int m = batch_size, n = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, incx = 1, incy = 1;
        double* a = static_cast<double*>(output_grad.data());
        double* x = ones.data();
        double* y = static_cast<double*>(bias_grad.data());
        dgemv_(
            &trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy
        );
    }
}

void op_nonzero(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    // TODO: not parallelized for now
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor output_tmp(DataType::dt_int, {batch_size});
    auto input_view = input.view<double, 1>();
    auto output_view = output_tmp.view<int64_t, 1>();
    std::size_t count = 0;
    for (std::size_t i = 0; i < batch_size; ++i) {
        if (input_view[i] != 0.) {
            output_view[count] = i;
            ++count;
        }
    }
    output = output_tmp.slice(0, 0, count);
}

template<int dim>
void batch_gather_impl(
    Tensor& indices, Tensor& values, Tensor& selection, const CpuDevice& device
) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;
    auto indices_view = indices.view<int64_t, 1>();
    if (values.dtype() == DataType::dt_float) {
        selection = Tensor(DataType::dt_float, out_shape, device);
        auto values_view = values.view<double, dim>();
        auto selection_view = selection.view<double, dim>();
        ThreadPool::instance().parallel_for([&](std::size_t i) {
            recursive_for<kernel_copy<CpuTypes>, dim-1>(
                values_view[indices_view[i]], selection_view[i]
            );
        }, batch_size);
    } else if (values.dtype() == DataType::dt_int) {
        selection = Tensor(DataType::dt_int, out_shape, device);
        auto values_view = values.view<int64_t, dim>();
        auto selection_view = selection.view<int64_t, dim>();
        ThreadPool::instance().parallel_for([&](std::size_t i) {
            recursive_for<kernel_copy_int<CpuTypes>, dim-1>(
                values_view[indices_view[i]], selection_view[i]
            );
        }, batch_size);
    } else {
        throw std::runtime_error("invalid dtype in batch_gather");
    }
}

void op_batch_gather(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    switch (values.shape().size()) {
        case 1: batch_gather_impl<1>(indices, values, selection, device); break;
        case 2: batch_gather_impl<2>(indices, values, selection, device); break;
        case 3: batch_gather_impl<3>(indices, values, selection, device); break;
        case 4: batch_gather_impl<4>(indices, values, selection, device); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template<int dim>
void scatter_impl(Tensor& indices, Tensor& source, Tensor& output) {
    auto batch_size = indices.size(0);
    auto indices_view = indices.view<int64_t, 1>();
    auto source_view = source.view<double, dim>();
    auto output_view = output.view<double, dim>();
    ThreadPool::instance().parallel_for([&](std::size_t i) {
        recursive_for<kernel_copy<CpuTypes>, dim-1>(
            source_view[i], output_view[indices_view[i]]
        );
    }, batch_size);
}

void op_scatter(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy(device);
    switch (target.shape().size()) {
        case 1: scatter_impl<1>(indices, source, output); break;
        case 2: scatter_impl<2>(indices, source, output); break;
        case 3: scatter_impl<3>(indices, source, output); break;
        case 4: scatter_impl<4>(indices, source, output); break;
        default:
            throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

void op_random(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(DataType::dt_float, {batch_size, dim}, device);
    auto output_view = output.view<double, 1>();
    auto& pool = ThreadPool::instance();
    pool.parallel_for<ThreadPool::pass_thread_id>(
        [&](std::size_t i, std::size_t thread_id
    ) {
        output_view[i] = pool.random(thread_id);
    }, batch_size * dim);
}

void op_unweight(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const CpuDevice& device
) {
    // TODO: not parallelized for now
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];

    auto batch_size = weights.size(0);
    Tensor indices_tmp(DataType::dt_int, {batch_size});
    Tensor uw_weights_tmp(DataType::dt_float, {batch_size});

    auto weights_view = weights.view<double, 1>();
    auto max_weight_view = max_weight.view<double, 1>();
    auto indices_view = indices_tmp.view<int64_t, 1>();
    auto uw_weights_view = uw_weights_tmp.view<double, 1>();
    auto& pool = ThreadPool::instance();

    std::size_t count = 0;
    for (std::size_t i = 0; i < batch_size; ++i) {
        double w = weights_view[i], w_max = max_weight_view[i];
        if (w != 0. && w > pool.random(0) * w_max) {
            indices_view[count] = i;
            uw_weights_view[count] = w > w_max ? w : w_max;
            ++count;
        }
    }

    indices = indices_tmp.slice(0, 0, count);
    uw_weights = uw_weights_tmp.slice(0, 0, count);
}

}

CpuRuntime::CpuRuntime(const Function& function, ContextPtr context) :
    context(context), input_count(function.inputs().size())
{
    locals_init.resize(function.locals().size());
    requires_grad_init.resize(function.locals().size());
    std::size_t instr_index = 0;
    LastUseOfLocals last_use(function);

    for (auto& instr : function.instructions()) {
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

    for (auto& [name, value] : function.globals()) {
        Tensor global = context->global(name);
        auto& global_shape = value.type.shape;
        Sizes full_shape(global_shape.size() + 1);
        full_shape[0] = 1;
        std::copy(global_shape.begin(), global_shape.end(), full_shape.begin() + 1);
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            throw std::invalid_argument(std::format(
                "Global {} has wrong dtype or shape", name
            ));
        }
        locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            requires_grad_init.at(value.local_index) = true;
            grad_global_indices.push_back({name, value.local_index});
        }
    }

    for (auto& local : function.locals()) {
        std::visit(Overloaded{
            [&](auto val) {
                Tensor tensor(val, &CpuDevice::instance());
                locals_init[local.local_index] = tensor;
            },
            [](std::monostate val){}
        }, local.literal_value);
    }

    for (auto& out : function.outputs()) {
        output_indices.push_back(out.local_index);
    }
}

TensorVec CpuRuntime::run(const TensorVec& inputs) const {
    auto& device = CpuDevice::instance();
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        switch (instr.opcode) {
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
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CpuRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto& device = CpuDevice::instance();
    auto locals = locals_init;
    auto requires_grad = requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    std::copy(input_requires_grad.begin(), input_requires_grad.end(), requires_grad.begin());

    for (auto [instr, instr_eval_grad] : zip(instructions, eval_grad)) {
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
    return {outputs, locals, eval_grad};
}

std::tuple<
    TensorVec, std::vector<std::tuple<std::string, Tensor>>
> CpuRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) const {
    auto& device = CpuDevice::instance();
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
        for (auto [output_index, output_dtype] : zip(
            instr.output_indices, instr.output_dtypes
        )) {
            auto& grad = local_grads[output_index];
            if (!grad && output_dtype == DataType::dt_float) {
                grad = Tensor(DataType::dt_float, locals[output_index].shape(), device);
                grad.zero(device);
            }
        }
        switch (instr.opcode) {
#include "runtime_backward_mixin.h"
        }
    }
    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    return {{local_grads.begin(), local_grads.begin() + input_count}, global_grads};
}

extern "C" Runtime* build_runtime(const Function& function, ContextPtr context) {
    return new CpuRuntime(function, context);
}

