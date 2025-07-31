#include "madevent/phasespace/base.h"

#include "madevent/util.h"

using namespace madevent;

namespace {

void check_types(
    const ValueVec& values, const TypeVec& types, const std::string& prefix
) {
    if (values.size() != types.size()) {
        throw std::runtime_error(std::format(
            "{}: Invalid number of values. Expected {}, got {}",
            prefix, types.size(), values.size()
        ));
    }
    std::size_t val_index = 1;
    for (auto [value, type] : zip(values, types)) {
        if (value.type.dtype != type.dtype) {
            throw std::runtime_error(std::format(
                "{}, value {}: Invalid dtype", prefix, val_index
            ));
        }
        if (value.type.shape != type.shape) {
            throw std::runtime_error(std::format(
                "{}, value {}: Invalid shape", prefix, val_index
            ));
        }
        ++val_index;
    }
}

}

Mapping::Result Mapping::build_forward(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    check_types(inputs, _input_types, "Input");
    check_types(conditions, _condition_types, "Condition");
    auto [outputs, det] = build_forward_impl(fb, inputs, conditions);
    check_types(outputs, _output_types, "Output");
    check_types({det}, {batch_float}, "Determinant");
    return {outputs, det};
}

Mapping::Result Mapping::build_inverse(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    check_types(inputs, _output_types, "Input");
    check_types(conditions, _condition_types, "Condition");
    auto [outputs, det] = build_inverse_impl(fb, inputs, conditions);
    check_types(outputs, _input_types, "Output");
    check_types({det}, {batch_float}, "Determinant");
    return {outputs, det};
}

Function Mapping::forward_function() const {
    auto arg_types = _input_types;
    arg_types.insert(arg_types.end(), _condition_types.begin(), _condition_types.end());
    auto ret_types = _output_types;
    ret_types.push_back(batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = _input_types.size();
    auto n_outputs = _output_types.size();
    auto [outputs, det] = build_forward_impl(
        fb, fb.input_range(0, n_inputs), fb.input_range(n_inputs, arg_types.size())
    );
    fb.output_range(0, outputs);
    fb.output(n_outputs, det);
    return fb.function();
}

Function Mapping::inverse_function() const {
    auto arg_types = _output_types;
    arg_types.insert(arg_types.end(), _condition_types.begin(), _condition_types.end());
    auto ret_types = _input_types;
    ret_types.push_back(batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = _input_types.size();
    auto n_outputs = _output_types.size();
    auto [outputs, det] = build_inverse_impl(
        fb, fb.input_range(0, n_outputs), fb.input_range(n_outputs, arg_types.size())
    );
    fb.output_range(0, outputs);
    fb.output(n_inputs, det);
    return fb.function();
}

ValueVec FunctionGenerator::build_function(
    FunctionBuilder& fb, const ValueVec& args
) const {
    check_types(args, _arg_types, "Argument");
    auto outputs = build_function_impl(fb, args);
    check_types(outputs, _return_types, "Output");
    return outputs;
}

Function FunctionGenerator::function() const {
    FunctionBuilder fb(_arg_types, _return_types);
    auto outputs = build_function_impl(fb, fb.input_range(0, _arg_types.size()));
    fb.output_range(0, outputs);
    return fb.function();
}
