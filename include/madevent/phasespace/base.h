#pragma once

#include "madevent/madcode.h"

namespace madevent {

class Mapping {
public:
    using Result = std::tuple<ValueVec, Value>;

    Mapping(
        const TypeVec& input_types,
        const TypeVec& output_types,
        const TypeVec& condition_types
    ) : _input_types(input_types),
        _output_types(output_types),
        _condition_types(condition_types) {}
    virtual ~Mapping() = default;
    Result build_forward(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const;
    Result build_inverse(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const;
    Function forward_function() const;
    Function inverse_function() const;
    const TypeVec& input_types() const { return _input_types; }
    const TypeVec& output_types() const { return _output_types; }
    const TypeVec& condition_types() const { return _condition_types; }

protected:
    //TODO: make parameters const ref
    virtual Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const = 0;
    virtual Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const = 0;

private:
    TypeVec _input_types;
    TypeVec _output_types;
    TypeVec _condition_types;
};

class FunctionGenerator {
public:
    FunctionGenerator(const TypeVec& arg_types, const TypeVec& return_types) :
        _arg_types(arg_types), _return_types(return_types) {}
    virtual ~FunctionGenerator() = default;
    ValueVec build_function(FunctionBuilder& fb, const ValueVec& args) const;
    Function function() const;
    const TypeVec& arg_types() const { return _arg_types; }
    const TypeVec& return_types() const { return _return_types; }

protected:
    virtual ValueVec build_function_impl(
        FunctionBuilder& fb, const ValueVec& args
    ) const = 0;

private:
    TypeVec _arg_types;
    TypeVec _return_types;
};

}
