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

using TensorValue = std::tuple<
    std::vector<int>, std::variant<std::vector<bool>, std::vector<long long>, std::vector<double>>
>;

using LiteralValue = std::variant<bool, long long, double, TensorValue, std::string, std::monostate>;

struct Value {
    Type type;
    LiteralValue literal_value;
    int local_index = -1;

    Value() : type(single_float), literal_value(std::monostate{}) {}

    Value(bool value) : type(single_bool), literal_value(value) {}
    Value(long long value) : type(single_int), literal_value(value) {}
    Value(double value) : type(single_float), literal_value(value) {}
    Value(std::string key, Type _type) : type(_type), literal_value(key) {}

    template<typename T>
    Value(const std::vector<T>& values, const std::vector<int>& shape = {}) :
        type{
            std::is_same_v<T, bool> ? DataType::dt_bool :
            std::is_same_v<T, long long> ? DataType::dt_int : DataType::dt_float,
            BatchSize::one,
            shape.size() == 0 ? std::vector<int>{static_cast<int>(values.size())} : shape
        },
        literal_value(TensorValue(type.shape, values))
    {
        std::size_t prod = 1;
        for (auto size : type.shape) {
            prod *= size;
        }
        if (prod != values.size()) {
            throw std::invalid_argument("size of value vector not compatible with given shape");
        }
    }

    Value(Type _type, int _local_index)
        : type(_type), literal_value(std::monostate{}), local_index(_local_index) {}
    Value(Type _type, LiteralValue _literal_value, int _local_index = -1)
        : type(_type), literal_value(_literal_value), local_index(_local_index) {}
};

using ValueList = std::vector<Value>;

struct InstructionCall {
    InstructionPtr instruction;
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
    FunctionBuilder(const Function& function);
    Value input(int index);
    ValueList input_range(int start_index, int end_index);
    void output(int index, Value value);
    void output_range(int start_index, const ValueList& values);
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
    std::vector<InstructionCall> instructions;
};

}
