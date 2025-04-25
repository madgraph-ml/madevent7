#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

constexpr std::size_t max_instr = 150;

template<typename T, std::size_t n>
struct AutogradArray {
    KERNELSPEC T& operator[](std::size_t i) { return arr[i]; }
    KERNELSPEC const T& operator[](std::size_t i) const { return arr[i]; }
    KERNELSPEC void fill(const T& value) {
        for (std::size_t i = 0; i < n; ++i) arr[i] = value;
    }
    T arr[n];
};

enum class AutogradOp {
    nop, load, store, literal, assign, copy, where,
    eq, neq, gt, lt, ge, le, band, bor, bnot,
    neg, add, sub, mul, div,
    sqrt, sin, cos, sinh, cosh, atan2, pow, fabs, log, tan, atan, exp, log1p
};

union AutogradScalar {
    double f;
    bool b;
    //int64_t i;
    KERNELSPEC AutogradScalar() {};
    KERNELSPEC AutogradScalar(double val) : f(val) {}
    KERNELSPEC AutogradScalar(bool val) : b(val) {}
    //KERNELSPEC AutogradScalar(int64_t val) : i(val) {}
};

template<typename T>
union AutogradEvalScalar {
    FVal<T> f;
    BVal<T> b;
    //IVal<T> i;
    KERNELSPEC AutogradEvalScalar() = default;
    KERNELSPEC AutogradEvalScalar(double val) : f(val) {}
    KERNELSPEC AutogradEvalScalar(bool val) : b(val) {}
    //KERNELSPEC AutogradEvalScalar(int64_t val) : i(val) {}
};

struct AutogradInstruction {
    AutogradOp op = AutogradOp::nop;
    AutogradScalar literal = 0.0;
    std::size_t arg1 = 0;
    std::size_t arg2 = 0;
    std::size_t arg3 = 0;
};

struct Graph {
    AutogradArray<AutogradInstruction, max_instr> instructions;
    std::size_t size = 0;

    template<typename... V>
    KERNELSPEC void append(V... args) {
        instructions[size] = {args...};
        ++size;
    }
};

template<typename T>
struct AutogradInput {
    KERNELSPEC AutogradInput(Graph& _graph, std::size_t _index) :
        graph(_graph), index(_index) {}

    Graph& graph;
    std::size_t index;
};

template<typename T>
struct AutogradValue {
    KERNELSPEC AutogradValue(AutogradInput<T> input) :
        graph(&input.graph), index(input.graph.size)
    {
        graph->append(AutogradOp::load, AutogradScalar{}, input.index);
    }

    KERNELSPEC AutogradValue(T literal) : graph(nullptr), index(0), literal(literal) {}

    KERNELSPEC AutogradValue(AutogradValue& val) : graph(val.graph) {
        if (graph == nullptr) {
            literal = val.literal;
        } else {
            index = graph->size;
            graph->append(AutogradOp::copy, AutogradScalar{}, val.index);
        }
    }

    KERNELSPEC AutogradValue(AutogradValue&& val) = default;

    KERNELSPEC AutogradValue(Graph* _graph, std::size_t _index) :
        graph(_graph), index(_index) {};

    KERNELSPEC AutogradValue(Graph* _graph, std::size_t _index, AutogradScalar _literal) :
        graph(_graph), index(_index), literal(_literal) {};

    template<typename... V>
    KERNELSPEC AutogradValue(AutogradOp op, V&... args) {
        ([&]{
            if (args.graph != nullptr) {
                graph = args.graph;
            }
        }(), ...);
        ([&]{
            if (args.graph == nullptr) {
                args.graph = graph;
                args.index = graph->size;
                graph->append(AutogradOp::literal, args.literal);
            }
        }(), ...);
        index = graph->size;
        graph->append(op, AutogradScalar{}, args.index...);
    }

    KERNELSPEC AutogradValue<T> operator=(AutogradValue<T> arg) {
        return {AutogradOp::assign, *this, arg};
    }

    Graph* graph;
    std::size_t index;
    AutogradScalar literal;
};

template<typename T>
struct AutogradOutput {
    KERNELSPEC AutogradOutput<T>& operator=(AutogradValue<T> value) {
        graph.append(AutogradOp::store, AutogradScalar{}, index, value.index);
        return *this;
    }

    Graph& graph;
    std::size_t index;
};

inline KERNELSPEC AutogradValue<double> where(
    AutogradValue<bool> arg1, AutogradValue<double> arg2, AutogradValue<double> arg3
) {
    return {AutogradOp::where, arg1, arg2, arg3};
}

inline KERNELSPEC AutogradValue<bool> operator==(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::eq, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator!=(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::neq, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator>(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::gt, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator<(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::lt, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator>=(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::ge, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator<=(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::le, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator&(AutogradValue<bool> arg1, AutogradValue<bool> arg2) {
    return {AutogradOp::band, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator|(AutogradValue<bool> arg1, AutogradValue<bool> arg2) {
    return {AutogradOp::bor, arg1, arg2};
}

inline KERNELSPEC AutogradValue<bool> operator!(AutogradValue<bool> arg1) {
    return {AutogradOp::bnot, arg1};
}

inline KERNELSPEC AutogradValue<double> operator-(AutogradValue<double> arg1) {
    return {AutogradOp::neg, arg1};
}

inline KERNELSPEC AutogradValue<double> operator+(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::add, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> operator-(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::sub, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> operator*(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::mul, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> operator/(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::div, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> sqrt(AutogradValue<double> arg1) {
    return {AutogradOp::sqrt, arg1};
}

inline KERNELSPEC AutogradValue<double> sin(AutogradValue<double> arg1) {
    return {AutogradOp::sin, arg1};
}

inline KERNELSPEC AutogradValue<double> cos(AutogradValue<double> arg1) {
    return {AutogradOp::cos, arg1};
}

inline KERNELSPEC AutogradValue<double> sinh(AutogradValue<double> arg1) {
    return {AutogradOp::sinh, arg1};
}

inline KERNELSPEC AutogradValue<double> cosh(AutogradValue<double> arg1) {
    return {AutogradOp::cosh, arg1};
}

inline KERNELSPEC AutogradValue<double> atan2(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::atan2, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> pow(AutogradValue<double> arg1, AutogradValue<double> arg2) {
    return {AutogradOp::pow, arg1, arg2};
}

inline KERNELSPEC AutogradValue<double> fabs(AutogradValue<double> arg1) {
    return {AutogradOp::fabs, arg1};
}

inline KERNELSPEC AutogradValue<double> log(AutogradValue<double> arg1) {
    return {AutogradOp::log, arg1};
}

inline KERNELSPEC AutogradValue<double> tan(AutogradValue<double> arg1) {
    return {AutogradOp::tan, arg1};
}

inline KERNELSPEC AutogradValue<double> atan(AutogradValue<double> arg1) {
    return {AutogradOp::atan, arg1};
}

inline KERNELSPEC AutogradValue<double> exp(AutogradValue<double> arg1) {
    return {AutogradOp::exp, arg1};
}

inline KERNELSPEC AutogradValue<double> log1p(AutogradValue<double> arg1) {
    return {AutogradOp::log1p, arg1};
}

template<typename T, const Graph& graph, std::size_t instr_index, std::size_t in_arg_count>
inline KERNELSPEC void eval_rec(
    AutogradArray<AutogradEvalScalar<T>, max_instr>& locals,
    AutogradArray<FIn<T,0>, in_arg_count> in_args
) {
    if constexpr (instr_index < max_instr) {
        auto& instr = graph.instructions[instr_index];
        switch (instr.op) {
        case AutogradOp::nop:
            break;
        case AutogradOp::load:
            locals[instr_index].f = in_args[instr.arg1];
            break;
        case AutogradOp::store:
            // *args[instr.arg1] = locals[instr.arg2].f;
            break;
        case AutogradOp::literal:
            locals[instr_index].f = instr.literal.f;
            break;
        case AutogradOp::assign:
            locals[instr.arg1] = locals[instr.arg2];
            locals[instr_index] = locals[instr.arg2];
            break;
        case AutogradOp::copy:
            locals[instr_index] = locals[instr.arg1];
            break;
        case AutogradOp::where:
            locals[instr_index].f = where(
                locals[instr.arg1].b, locals[instr.arg2].f, locals[instr.arg3].f
            );
            break;
        case AutogradOp::eq:
            locals[instr_index].b = locals[instr.arg1].f == locals[instr.arg2].f;
            break;
        case AutogradOp::neq:
            locals[instr_index].b = locals[instr.arg1].f != locals[instr.arg2].f;
            break;
        case AutogradOp::gt:
            locals[instr_index].b = locals[instr.arg1].f > locals[instr.arg2].f;
            break;
        case AutogradOp::lt:
            locals[instr_index].b = locals[instr.arg1].f < locals[instr.arg2].f;
            break;
        case AutogradOp::ge:
            locals[instr_index].b = locals[instr.arg1].f >= locals[instr.arg2].f;
            break;
        case AutogradOp::le:
            locals[instr_index].b = locals[instr.arg1].f <= locals[instr.arg2].f;
            break;
        case AutogradOp::band:
            locals[instr_index].b = locals[instr.arg1].b & locals[instr.arg2].b;
            break;
        case AutogradOp::bor:
            locals[instr_index].b = locals[instr.arg1].b | locals[instr.arg2].b;
            break;
        case AutogradOp::bnot:
            locals[instr_index].b = ! locals[instr.arg1].b;
            break;
        case AutogradOp::neg:
            locals[instr_index].f = - locals[instr.arg1].f;
            break;
        case AutogradOp::add:
            locals[instr_index].f = locals[instr.arg1].f + locals[instr.arg2].f;
            break;
        case AutogradOp::sub:
            locals[instr_index].f = locals[instr.arg1].f - locals[instr.arg2].f;
            break;
        case AutogradOp::mul:
            locals[instr_index].f = locals[instr.arg1].f * locals[instr.arg2].f;
            break;
        case AutogradOp::div:
            locals[instr_index].f = locals[instr.arg1].f / locals[instr.arg2].f;
            break;
        case AutogradOp::sqrt:
            locals[instr_index].f = sqrt(locals[instr.arg1].f);
            break;
        case AutogradOp::sin:
            locals[instr_index].f = sin(locals[instr.arg1].f);
            break;
        case AutogradOp::cos:
            locals[instr_index].f = cos(locals[instr.arg1].f);
            break;
        case AutogradOp::sinh:
            locals[instr_index].f = sinh(locals[instr.arg1].f);
            break;
        case AutogradOp::cosh:
            locals[instr_index].f = cosh(locals[instr.arg1].f);
            break;
        case AutogradOp::atan2:
            locals[instr_index].f = atan2(locals[instr.arg1].f, locals[instr.arg2].f);
            break;
        case AutogradOp::pow:
            locals[instr_index].f = pow(locals[instr.arg1].f, locals[instr.arg2].f);
            break;
        case AutogradOp::fabs:
            locals[instr_index].f = fabs(locals[instr.arg1].f);
            break;
        case AutogradOp::log:
            locals[instr_index].f = log(locals[instr.arg1].f);
            break;
        case AutogradOp::tan:
            locals[instr_index].f = tan(locals[instr.arg1].f);
            break;
        case AutogradOp::atan:
            locals[instr_index].f = atan(locals[instr.arg1].f);
            break;
        case AutogradOp::exp:
            locals[instr_index].f = exp(locals[instr.arg1].f);
            break;
        case AutogradOp::log1p:
            locals[instr_index].f = log1p(locals[instr.arg1].f);
            break;
        }
        eval_rec<T, graph, instr_index + 1, in_arg_count>(locals, in_args);
    }
}

template<typename U, typename V>
inline KERNELSPEC void accumulate_grad(bool& is_init, U& grad, V val) {
    grad = is_init ? grad + val : val;
    is_init = true;
}

template<
    typename T,
    const Graph& graph,
    std::size_t rev_instr_index,
    std::size_t in_arg_count,
    std::size_t out_arg_count
>
__attribute__((always_inline))
inline KERNELSPEC void backward_rec(
    AutogradArray<AutogradEvalScalar<T>, max_instr>& locals,
    AutogradArray<AutogradEvalScalar<T>, max_instr>& local_grads,
    AutogradArray<bool, max_instr>& local_grads_init,
    AutogradArray<FIn<T,0>, in_arg_count>& in_args,
    AutogradArray<FOut<T,0>, in_arg_count>& in_grads,
    AutogradArray<FIn<T,0>, out_arg_count>& out_grads
) {
    if constexpr (rev_instr_index < max_instr) {
        auto instr_index = max_instr - rev_instr_index - 1;
        auto& instr = graph.instructions[instr_index];
        auto grad = local_grads[instr_index].f;
        switch (instr.op) {
        case AutogradOp::nop:
            break;
        case AutogradOp::load:
            in_grads[instr.arg1] = in_grads[instr.arg1] + grad;
            break;
        case AutogradOp::store:
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                out_grads[instr.arg1]
            );
            break;
        case AutogradOp::literal:
            break;
        case AutogradOp::assign:
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                local_grads[instr.arg1].f
            );
            accumulate_grad(
                local_grads_init[instr.arg2], local_grads[instr.arg2].f, grad
            );
            break;
        case AutogradOp::copy:
            accumulate_grad(
                local_grads_init[instr.arg1], local_grads[instr.arg1].f, grad
            );
            break;
        case AutogradOp::where: {
            auto cond = locals[instr.arg1].b;
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                where(cond, grad, 0.0)
            );
            accumulate_grad(
                local_grads_init[instr.arg3],
                local_grads[instr.arg3].f,
                where(cond, 0.0, grad)
            );
            break;
        } case AutogradOp::eq:
        case AutogradOp::neq:
        case AutogradOp::gt:
        case AutogradOp::lt:
        case AutogradOp::ge:
        case AutogradOp::le:
        case AutogradOp::band:
        case AutogradOp::bor:
        case AutogradOp::bnot:
            accumulate_grad(local_grads_init[instr.arg1], local_grads[instr.arg1].f, 0.0);
            accumulate_grad(local_grads_init[instr.arg2], local_grads[instr.arg2].f, 0.0);
            break;
        case AutogradOp::neg:
            accumulate_grad(
                local_grads_init[instr.arg1], local_grads[instr.arg1].f, -grad
            );
            break;
        case AutogradOp::add:
            accumulate_grad(
                local_grads_init[instr.arg1], local_grads[instr.arg1].f, grad
            );
            accumulate_grad(
                local_grads_init[instr.arg2], local_grads[instr.arg2].f, grad
            );
            break;
        case AutogradOp::sub:
            accumulate_grad(
                local_grads_init[instr.arg1], local_grads[instr.arg1].f, grad
            );
            accumulate_grad(
                local_grads_init[instr.arg2], local_grads[instr.arg2].f, -grad
            );
            break;
        case AutogradOp::mul:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * locals[instr.arg2].f
            );
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                grad * locals[instr.arg1].f
            );
            break;
        case AutogradOp::div: {
            auto x = locals[instr.arg1].f, y = locals[instr.arg2].f;
            accumulate_grad(
                local_grads_init[instr.arg1], local_grads[instr.arg1].f, grad / y
            );
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                - grad * locals[instr_index].f / y
            );
            break;
        } case AutogradOp::sqrt:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                0.5 * grad / locals[instr_index].f
            );
            break;
        case AutogradOp::sin:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * cos(locals[instr.arg1].f)
            );
            break;
        case AutogradOp::cos:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                - grad * sin(locals[instr.arg1].f)
            );
            break;
        case AutogradOp::sinh:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * cosh(locals[instr.arg1].f)
            );
            break;
        case AutogradOp::cosh:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * sinh(locals[instr.arg1].f)
            );
            break;
        case AutogradOp::atan2: {
            auto y = locals[instr.arg1].f, x = locals[instr.arg2].f;
            auto r2 = x * x + y * y;
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * x / r2
            );
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                - grad * y / r2
            );
            break;
        } case AutogradOp::pow: {
            auto x = locals[instr.arg1].f, y = locals[instr.arg2].f;
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * y * pow(x, y - 1)
            );
            accumulate_grad(
                local_grads_init[instr.arg2],
                local_grads[instr.arg2].f,
                grad * log(x) * locals[instr_index].f
            );
            break;
        } case AutogradOp::fabs:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * locals[instr.arg1].f / locals[instr_index].f
            );
            break;
        case AutogradOp::log:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad / locals[instr.arg1].f
            );
            break;
        case AutogradOp::tan: {
            auto cos_x = cos(locals[instr.arg1].f);
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad / (cos_x * cos_x)
            );
            break;
        } case AutogradOp::atan: {
            auto x = locals[instr.arg1].f;
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad / (x * x + 1)
            );
            break;
        } case AutogradOp::exp:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad * exp(locals[instr.arg1].f)
            );
            break;
        case AutogradOp::log1p:
            accumulate_grad(
                local_grads_init[instr.arg1],
                local_grads[instr.arg1].f,
                grad / (locals[instr.arg1].f + 1)
            );
            break;
        }
        backward_rec<T, graph, rev_instr_index + 1, in_arg_count, out_arg_count>(
            locals, local_grads, local_grads_init, in_args, in_grads, out_grads
        );
    }
}

template<std::size_t... i>
inline KERNELSPEC auto in_args(Graph& graph, std::index_sequence<i...>) {
    return std::make_tuple(AutogradInput<double>(graph, i)...);
}

template<std::size_t... i>
inline KERNELSPEC auto out_args(Graph& graph, std::index_sequence<i...>) {
    return std::make_tuple(AutogradOutput<double>(graph, i)...);
}

template<auto func, int in_arg_count, int out_arg_count>
inline KERNELSPEC Graph func_graph() {
    Graph graph;
    std::apply(
        func,
        std::tuple_cat(
            in_args(graph, std::make_index_sequence<in_arg_count>{}),
            out_args(graph, std::make_index_sequence<out_arg_count>{})
        )
    );
    return graph;
}

template<auto func, int in_arg_count, int out_arg_count>
KERNELSPEC Graph autograd_graph = func_graph<func, in_arg_count, out_arg_count>();

template<typename T, auto func, int in_arg_count, int out_arg_count>
inline KERNELSPEC void backward(
    AutogradArray<FIn<T,0>, in_arg_count> input_args,
    AutogradArray<FIn<T,0>, out_arg_count> output_grads,
    AutogradArray<FOut<T,0>, in_arg_count> input_grads
) {
    //static constexpr Graph graph = func_graph<func, in_arg_count, out_arg_count>();
    AutogradArray<AutogradEvalScalar<T>, max_instr> locals;
    AutogradArray<AutogradEvalScalar<T>, max_instr> local_grads;
    AutogradArray<bool, max_instr> local_grads_init;
    local_grads_init.fill(false);
    eval_rec<T, autograd_graph<func, in_arg_count, out_arg_count>, 0, in_arg_count>(locals, input_args);
    backward_rec<T, autograd_graph<func, in_arg_count, out_arg_count>, 0, in_arg_count, out_arg_count>(
        locals, local_grads, local_grads_init, input_args, input_grads, output_grads
    );
}

struct AutogradTypes {
    template<int dim> using FIn = AutogradInput<double>;
    //template<int dim> using IIn = const VectorizedTensorView<IVec, int64_t, dim, false>;
    //template<int dim> using BIn = const VectorizedTensorView<BVec, bool, dim, false>;
    template<int dim> using FOut = AutogradOutput<double>;
    //template<int dim> using IOut = VectorizedTensorView<IVec, int64_t, dim, false>;
    //template<int dim> using BOut = VectorizedTensorView<BVec, bool, dim, false>;
    using FVal = AutogradValue<double>;
    //using IVal = AutogradValue<int64_t>;
    using BVal = AutogradValue<bool>;
};

}
}
