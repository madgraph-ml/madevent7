#pragma once

#include "madevent/madcode.h"

namespace madevent {

class Mapping {
public:
    using Result = std::tuple<ValueVec, Value>;

    Mapping(
        TypeVec _input_types,
        TypeVec _output_types,
        TypeVec _condition_types
    ) : input_types(_input_types),
        output_types(_output_types),
        condition_types(_condition_types) {}
    virtual ~Mapping() = default;
    Result build_forward(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const;
    Result build_inverse(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const;
    Function forward_function() const;
    Function inverse_function() const;
    const TypeVec& get_input_types() const { return input_types; }
    const TypeVec& get_output_types() const { return output_types; }
    const TypeVec& get_condition_types() const { return condition_types; }

protected:
    TypeVec input_types;
    TypeVec output_types;
    TypeVec condition_types;

    virtual Result build_forward_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const = 0;
    virtual Result build_inverse_impl(
        FunctionBuilder& fb, ValueVec inputs, ValueVec conditions
    ) const = 0;
    void check_types(ValueVec values, TypeVec types, std::string prefix) const;
};

class FunctionGenerator {
public:
    FunctionGenerator(const TypeVec& arg_types, const TypeVec& return_types) :
        _arg_types(arg_types), _return_types(return_types) {}
    virtual ~FunctionGenerator() = default;
    ValueVec build_function(FunctionBuilder& fb, const ValueVec& args) const;
    Function function() const;

protected:
    TypeVec _arg_types;
    TypeVec _return_types;

    virtual ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const = 0;
    void check_types(
        const ValueVec& values, const TypeVec& types, const std::string& prefix
    ) const;
};

}
