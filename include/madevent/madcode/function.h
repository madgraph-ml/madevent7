#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <variant>
#include <string>
#include <type_traits>
#include <algorithm>
#include <optional>
#include <array>

#include "instruction.h"

namespace madevent {

using LiteralValue = std::variant<bool, int, double, std::string, std::monostate>;

struct Value {
    Type type;
    LiteralValue literal_value;
    int local_index = -1;

    Value() : type(scalar), literal_value(std::monostate{}) {}
    Value(bool value) : type(scalar_bool), literal_value(value) {}
    Value(int value) : type(scalar_int), literal_value(value) {}
    Value(double value) : type(scalar), literal_value(value) {}
    Value(std::string key, Type _type) : type(_type), literal_value(key) {}
    Value(Type _type, int _local_index)
        : type(_type), literal_value(std::monostate{}), local_index(_local_index) {}
    Value(Type _type, LiteralValue _literal_value, int _local_index)
        : type(_type), literal_value(_literal_value), local_index(_local_index) {}
};

using ValueList = std::vector<Value>;

struct InstructionCall {
    const InstructionPtr& instruction;
    ValueList inputs;
    ValueList outputs;
};

struct Function {
    ValueList inputs;
    ValueList outputs;
    ValueList locals;
    std::vector<InstructionCall> instructions;
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueList& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

class FunctionBuilder {
public:
    FunctionBuilder(const std::vector<Type> _input_types, const std::vector<Type> _output_types);
    Value input(int index);
    ValueList input_range(int start_index, int end_index);
    void output(int index, Value value);
    void output_range(int start_index, ValueList values);
    ValueList instruction(std::string name, ValueList args);
    Function function();

    Value sum(ValueList values, Value zero = 0.);
    Value product(ValueList values, Value one = 1.);

#include "function_builder_mixin.h"

private:
    std::vector<Type> output_types;
    ValueList inputs;
    std::vector<std::optional<Value>> outputs;
    std::map<LiteralValue, Value> literals;
    ValueList locals;
    std::vector<InstructionCall> instructions;
};

}
