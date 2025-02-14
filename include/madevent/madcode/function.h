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

struct InstructionCall {
    InstructionPtr instruction;
    ValueList inputs;
    ValueList outputs;
};

struct Function {
    ValueList inputs;
    ValueList outputs;
    ValueList locals;
    std::unordered_map<std::string, Value> globals;
    std::vector<InstructionCall> instructions;
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueList& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

class FunctionBuilder {
public:
    FunctionBuilder(const std::vector<Type> _input_types, const std::vector<Type> _output_types);
    FunctionBuilder(const Function& function);
    Value input(int index);
    ValueList input_range(int start_index, int end_index);
    void output(int index, Value value);
    void output_range(int start_index, const ValueList& values);
    Value global(std::string name, DataType dtype, const std::vector<int>& shape);
    ValueList instruction(std::string name, ValueList args);
    ValueList instruction(InstructionPtr instruction, ValueList args);
    Function function();

    Value sum(const ValueList& values);
    Value product(const ValueList& values);

#include "function_builder_mixin.h"

private:
    std::vector<Type> output_types;
    ValueList inputs;
    std::vector<std::optional<Value>> outputs;
    std::map<LiteralValue, Value> literals;
    ValueList locals;
    std::unordered_map<std::string, Value> globals;
    std::vector<InstructionCall> instructions;
};

}
