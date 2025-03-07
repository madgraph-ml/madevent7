#pragma once

#include "madevent/madcode.h"

namespace madevent {

class Mapping {
public:
    using Result = std::tuple<ValueList, Value>;

    Mapping(
        TypeList _input_types,
        TypeList _output_types,
        TypeList _condition_types
    ) : input_types(_input_types),
        output_types(_output_types),
        condition_types(_condition_types) {}
    virtual ~Mapping() = default;
    Result build_forward(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const;
    Result build_inverse(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const;
    Function forward_function() const;
    Function inverse_function() const;
    const TypeList& get_input_types() const { return input_types; }
    const TypeList& get_output_types() const { return output_types; }
    const TypeList& get_condition_types() const { return condition_types; }

protected:
    TypeList input_types;
    TypeList output_types;
    TypeList condition_types;

    virtual Result build_forward_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const = 0;
    virtual Result build_inverse_impl(
        FunctionBuilder& fb, ValueList inputs, ValueList conditions
    ) const = 0;
    void check_types(ValueList values, TypeList types, std::string prefix) const;
};

class FunctionGenerator {
public:
    FunctionGenerator(const TypeList& arg_types, const TypeList& return_types) :
        _arg_types(arg_types), _return_types(return_types) {}
    virtual ~FunctionGenerator() = default;
    ValueList build_function(FunctionBuilder& fb, const ValueList& args) const;
    Function function() const;

protected:
    TypeList _arg_types;
    TypeList _return_types;

    virtual ValueList build_function_impl(FunctionBuilder& fb, const ValueList& args) const = 0;
    void check_types(
        const ValueList& values, const TypeList& types, const std::string& prefix
    ) const;
};

}
