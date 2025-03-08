#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <string>
#include <optional>

#include <nlohmann/json.hpp>

#include "instruction.h"

namespace madevent {

struct InstructionCall {
    InstructionPtr instruction;
    ValueVec inputs;
    ValueVec outputs;
};

struct Function {
    ValueVec inputs;
    ValueVec outputs;
    ValueVec locals;
    std::unordered_map<std::string, Value> globals;
    std::vector<InstructionCall> instructions;

    void store(std::string file);
    static Function load(std::string file);
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueVec& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

void to_json(nlohmann::json& j, const InstructionCall& call);
void to_json(nlohmann::json& j, const Function& call);
void from_json(const nlohmann::json& j, Function& call);

class FunctionBuilder {
public:
    FunctionBuilder(
        const std::vector<Type> _input_types, const std::vector<Type> _output_types
    );
    FunctionBuilder(const Function& function);
    Value input(int index);
    ValueVec input_range(int start_index, int end_index);
    void output(int index, Value value);
    void output_range(int start_index, const ValueVec& values);
    Value global(std::string name, DataType dtype, const std::vector<int>& shape);
    ValueVec instruction(std::string name, ValueVec args);
    ValueVec instruction(InstructionPtr instruction, ValueVec args);
    Function function();

    Value sum(const ValueVec& values);
    //Value product(const ValueVec& values);

#include "function_builder_mixin.h"

private:
    std::vector<Type> output_types;
    ValueVec inputs;
    std::vector<std::optional<Value>> outputs;
    std::map<LiteralValue, Value> literals;
    ValueVec locals;
    std::unordered_map<std::string, Value> globals;
    std::vector<InstructionCall> instructions;
};

}
