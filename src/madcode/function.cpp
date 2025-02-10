#include "madevent/madcode/function.h"
#include "madevent/util.h"

#include <stdexcept>
#include <format>

using namespace madevent;

std::ostream& madevent::operator<<(std::ostream& out, const Value& value) {
    std::visit(Overloaded{
        [&out](auto val) { out << val; },
        [&out](TensorValue val) {
            std::visit([&out](auto items) {
                out << "{";
                std::size_t i = 0;
                for (auto item : items) {
                    if (i == items.size() - 1) {
                        out << item;
                    } else if (i == 20) {
                        out << "...";
                        break;
                    } else {
                        out << item << ", ";
                    }
                    ++i;
                }
                out << "}";
            }, std::get<1>(val));
        },
        [&out, value](std::monostate val) { out << "%" << value.local_index; }
    }, value.literal_value);
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const ValueList& list) {
    bool first = true;
    for (auto value : list) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << value;
    }
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const InstructionCall& call) {
    out << call.outputs << " = " << call.instruction->name << "(" << call.inputs << ")";
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const Function& func) {
    out << "inputs " << func.inputs << "\n";
    for (auto& instr : func.instructions) {
        out << instr << "\n";
    }
    out << "outputs " << func.outputs << "\n";
    return out;
}

FunctionBuilder::FunctionBuilder(
    const std::vector<Type> input_types, const std::vector<Type> _output_types
) : output_types(_output_types) {
    for (auto input_type : input_types) {
        Value value(input_type, locals.size());
        locals.push_back(value);
        inputs.push_back(value);
    }
    for (auto output_type : output_types) {
        outputs.push_back(std::nullopt);
    }
}

FunctionBuilder::FunctionBuilder(const Function& function) :
    inputs(function.inputs), locals(function.inputs)
{
    TypeList output_types;
    for (auto& output : function.outputs) {
        output_types.push_back(output.type);
        outputs.push_back(std::nullopt);
    }
}

ValueList FunctionBuilder::instruction(std::string name, ValueList args) {
    auto find_instr = instruction_set.find(name);
    if (find_instr == instruction_set.end()) {
        throw std::invalid_argument(std::format("Unknown instruction '{}'", name));
    }
    return instruction(find_instr->second.get(), args);
}

ValueList FunctionBuilder::instruction(InstructionPtr instruction, ValueList args) {
    int arg_index = -1;
    std::vector<Type> arg_types;
    for (auto& arg : args) {
        ++arg_index;
        arg_types.push_back(arg.type);

        if (arg.local_index != -1) {
            if (arg.local_index < 0 || arg.local_index > locals.size()) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: inconsistent value (local index)", instruction->name, arg_index
                ));
            }
            auto local_value = locals.at(arg.local_index);
            if (local_value.type != arg.type || local_value.literal_value != arg.literal_value) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: inconsistent value (type or value)", instruction->name, arg_index
                ));
            }
            continue;
        }
        if (std::holds_alternative<std::monostate>(arg.literal_value)) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: undefined value", instruction->name, arg_index
            ));
        }
        auto find_literal = literals.find(arg.literal_value);
        if (find_literal == literals.end()) {
            int local_index = locals.size();
            arg = Value(arg.type, arg.literal_value, local_index);
            locals.push_back(arg);
            literals[arg.literal_value] = arg;
        } else {
            arg = find_literal->second;
        }
    }

    auto output_types = instruction->signature(arg_types);
    ValueList call_outputs;
    for (const auto& type : output_types) {
        Value value(type, locals.size());
        locals.push_back(value);
        call_outputs.push_back(value);
    }
    instructions.push_back(InstructionCall{instruction, args, call_outputs});
    return call_outputs;
}

Function FunctionBuilder::function() {
    ValueList func_outputs;
    int output_index = 0;
    for (auto output : outputs) {
        if (output) {
            func_outputs.push_back(output.value());
        } else {
            throw std::invalid_argument(std::format(
                "No value assigned to output {}", output_index
            ));
        }
        ++output_index;
    }
    return Function{inputs, func_outputs, locals, instructions};
}

Value FunctionBuilder::input(int index) {
    if (index < 0 || index >= inputs.size()) {
        throw std::out_of_range(std::format(
            "Input index expected to be in range 0 to {}, got {}",
            inputs.size() - 1, index
        ));
    }
    return inputs.at(index);
}

ValueList FunctionBuilder::input_range(int start_index, int end_index) {
    if (start_index < 0 || start_index > inputs.size()) {
        throw std::out_of_range(std::format(
            "Start index expected to be in range 0 to {}, got {}",
            inputs.size(), start_index
        ));
    }
    if (end_index < start_index || end_index > inputs.size()) {
        throw std::out_of_range(std::format(
            "End index expected to be in range {} to {}, got {}",
            start_index, inputs.size() - 1, end_index
        ));
    }
    return ValueList(inputs.begin() + start_index, inputs.begin() + end_index);
}

void FunctionBuilder::output(int index, Value value) {
    if (index < 0 || index >= outputs.size()) {
        throw std::out_of_range(std::format(
            "Output index expected to be in range 0 to {}, got {}",
            outputs.size() - 1, index
        ));
    }
    if (output_types.at(index) != value.type) {
        throw std::invalid_argument(std::format("Wrong output type for output {}", index));
    }
    outputs.at(index) = value;
}

void FunctionBuilder::output_range(int start_index, const ValueList& values) {
    if (start_index < 0 || start_index > outputs.size() - values.size()) {
        throw std::out_of_range(std::format(
            "Start index expected to be in range 0 to {}, got {}",
            outputs.size() - values.size(), start_index
        ));
    }
    std::copy(values.begin(), values.end(), outputs.begin() + start_index);
}

Value FunctionBuilder::sum(const ValueList& values) {
    if (values.size() == 0) {
        return 0.0;
    }
    auto result = values.at(0);
    for (auto value = values.begin() + 1; value != values.end(); ++value) {
        result = add(result, *value);
    }
    return result;
}

Value FunctionBuilder::product(const ValueList& values) {
    if (values.size() == 0) {
        return 1.;
    }
    auto result = values.at(0);
    for (auto value = values.begin() + 1; value != values.end(); ++value) {
        result = mul(result, *value);
    }
    return result;
}
