#include "madevent/phasespace/mapping.h"

using namespace madevent;

Mapping::Result Mapping::build_forward(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    check_types(inputs, input_types, "Input");
    check_types(conditions, condition_types, "Condition");
    auto [outputs, det] = build_forward_impl(fb, inputs, conditions);
    check_types(outputs, output_types, "Output");
    check_types({det}, {batch_float}, "Determinant");
    return {outputs, det};
}

Mapping::Result Mapping::build_inverse(
    FunctionBuilder& fb, ValueList inputs, ValueList conditions
) const {
    check_types(inputs, output_types, "Input");
    check_types(conditions, condition_types, "Condition");
    auto [outputs, det] = build_inverse_impl(fb, inputs, conditions);
    check_types(outputs, input_types, "Output");
    check_types({det}, {batch_float}, "Determinant");
    return {outputs, det};
}

Function Mapping::forward_function() const {
    auto arg_types = input_types;
    arg_types.insert(arg_types.end(), condition_types.begin(), condition_types.end());
    auto ret_types = output_types;
    ret_types.push_back(batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = input_types.size();
    auto n_outputs = output_types.size();
    auto [outputs, det] = build_forward_impl(
        fb, fb.input_range(0, n_inputs), fb.input_range(n_inputs, arg_types.size())
    );
    fb.output_range(0, outputs);
    fb.output(n_outputs, det);
    return fb.function();
}

Function Mapping::inverse_function() const {
    auto arg_types = output_types;
    arg_types.insert(arg_types.end(), condition_types.begin(), condition_types.end());
    auto ret_types = input_types;
    ret_types.push_back(batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = input_types.size();
    auto n_outputs = output_types.size();
    auto [outputs, det] = build_inverse_impl(
        fb, fb.input_range(0, n_outputs), fb.input_range(n_outputs, arg_types.size())
    );
    fb.output_range(0, outputs);
    fb.output(n_inputs, det);
    return fb.function();
}

void Mapping::check_types(ValueList values, TypeList types, std::string prefix) const {
    
}
