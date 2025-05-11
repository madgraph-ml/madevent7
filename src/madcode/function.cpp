#include "madevent/madcode/function.h"
#include "madevent/util.h"

#include <stdexcept>
#include <format>
#include <fstream>

using namespace madevent;
using json = nlohmann::json;

void Function::store(const std::string& file) const {
    std::ofstream f(file);
    json j;
    j = *this;
    f << j.dump(2);
}

Function Function::load(const std::string& file) {
    std::ifstream f(file);
    return json::parse(f).get<Function>();
}

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

std::ostream& madevent::operator<<(std::ostream& out, const ValueVec& list) {
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
    out << call.outputs << " = " << call.instruction->name() << "(" << call.inputs << ")";
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const Function& func) {
    out << "Inputs:\n";
    for (auto& input : func.inputs()) {
        out << "  " << input << " : " << input.type << "\n";
    }
    out << "Globals:\n";
    for (auto& [name, global] : func.globals()) {
        out << "  " << global << " : " << global.type << " = " << name << "\n";
    }
    out << "Instructions:\n";
    for (auto& instr : func.instructions()) {
        out << "  " << instr << "\n";
    }
    out << "Outputs:\n";
    for (auto& output : func.outputs()) {
        out << "  " << output << " : " << output.type << "\n";
    }
    return out;
}

void madevent::to_json(json& j, const InstructionCall& call) {
    j = json{
        {"name", call.instruction->name()},
        {"inputs", call.inputs},
        {"outputs", call.outputs},
    };
}

void madevent::to_json(json& j, const Function& func) {
    json inputs(json::value_t::array);
    json outputs(json::value_t::array);
    json globals(json::value_t::array);
    for (auto& input : func.inputs()) {
        inputs.push_back(json{
            {"local", input},
            {"dtype", input.type.dtype},
            {"batch_size", input.type.batch_size},
            {"shape", input.type.shape},
        });
    }
    for (auto& output : func.outputs()) {
        outputs.push_back(json{
            {"local", output},
            {"dtype", output.type.dtype},
            {"shape", output.type.shape},
        });
    }
    for (auto& [name, global] : func.globals()) {
        globals.push_back(json{
            {"local", global},
            {"name", name},
            {"dtype", global.type.dtype},
            {"shape", global.type.shape},
        });
    }
    j = json{
        {"inputs", inputs},
        {"outputs", outputs},
        {"globals", globals},
        {"instructions", func.instructions()},
    };
}

void madevent::from_json(const json& j, Function& func) {
    TypeVec input_types, output_types;
    std::vector<std::size_t> input_locals, output_locals;
    for (auto& j_input : j.at("inputs")) {
        BatchSize batch_size = BatchSize::one;
        j_input.at("batch_size").get_to<BatchSize>(batch_size);
        input_types.emplace_back(
            j_input.at("dtype").get<DataType>(),
            batch_size,
            j_input.at("shape").get<std::vector<int>>()
        );
        input_locals.push_back(j_input.at("local").get<std::size_t>());
    }
    for (auto& j_output : j.at("outputs")) {
        output_types.emplace_back(
            j_output.at("dtype").get<DataType>(),
            BatchSize::one,
            j_output.at("shape").get<std::vector<int>>()
        );
        output_locals.push_back(j_output.at("local").get<std::size_t>());
    }

    FunctionBuilder fb(input_types, output_types);
    std::unordered_map<std::size_t, Value> locals;
    std::size_t i = 0;
    for (auto input_index : input_locals) {
        locals[input_index] = fb.input(i);
        ++i;
    }

    for (auto& j_global : j.at("globals")) {
        locals[j_global.at("local").get<std::size_t>()] = fb.global(
            j_global.at("name").get<std::string>(),
            j_global.at("dtype").get<DataType>(),
            j_global.at("shape").get<std::vector<int>>()
        );
    }

    for (auto& j_instr : j.at("instructions")) {
        ValueVec instr_inputs;
        for (auto& j_input : j_instr.at("inputs")) {
            instr_inputs.push_back(
                j_input.is_number_unsigned() ?
                locals.at(j_input.get<std::size_t>()) :
                j_input.get<Value>()
            );
        }
        auto instr_outputs = fb.instruction(
            j_instr.at("name").get<std::string>(), instr_inputs
        );
        for (auto [j_output, output] : zip(
            j_instr.at("outputs"), instr_outputs
        )) {
            locals[j_output.get<std::size_t>()] = output;
        }
    }

    i = 0;
    for (auto output_index : output_locals) {
        fb.output(i, locals[output_index]);
        ++i;
    }
    func = fb.function();
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
    inputs(function.inputs()), locals(function.inputs())
{
    TypeVec output_types;
    for (auto& output : function.outputs()) {
        output_types.push_back(output.type);
        outputs.push_back(std::nullopt);
    }
}

ValueVec FunctionBuilder::instruction(const std::string& name, const ValueVec& args) {
    auto find_instr = instruction_set.find(name);
    if (find_instr == instruction_set.end()) {
        throw std::invalid_argument(std::format("Unknown instruction '{}'", name));
    }
    return instruction(find_instr->second.get(), args);
}

ValueVec FunctionBuilder::instruction(InstructionPtr instruction, const ValueVec& args) {
    auto params = args;
    int arg_index = -1;
    bool const_opt = true;
    for (auto& arg : params) {
        ++arg_index;

        if (arg.local_index != -1) {
            if (arg.local_index < 0 || arg.local_index > locals.size()) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: inconsistent value (local index)",
                    instruction->name(), arg_index
                ));
            }
            auto local_value = locals.at(arg.local_index);
            if (local_value.type != arg.type ||
                local_value.literal_value != arg.literal_value) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: inconsistent value (type or value)",
                    instruction->name(), arg_index
                ));
            }
            const_opt = false;
        } else if (std::holds_alternative<std::monostate>(arg.literal_value)) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: undefined value", instruction->name(), arg_index
            ));
        }
    }

    if (const_opt) {
        switch (instruction->opcode()) {
        case opcodes::add: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 + arg1};
        } case opcodes::sub: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 - arg1};
        } case opcodes::mul: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 * arg1};
        } case opcodes::clip_min: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 < arg1 ? arg1 : arg0};
        } case opcodes::sqrt: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            return {std::sqrt(arg0)};
        } case opcodes::square: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            return {arg0 * arg0};
        } case opcodes::pow: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {std::pow(arg0, arg1)};
        }}
    }

    for (auto& arg : params) {
        if (arg.local_index != -1) continue;

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

    auto output_types = instruction->signature(params);
    ValueVec call_outputs;
    for (const auto& type : output_types) {
        Value value(type, locals.size());
        locals.push_back(value);
        call_outputs.push_back(value);
    }
    instructions.push_back(InstructionCall{instruction, params, call_outputs});
    return call_outputs;
}

Function FunctionBuilder::function() {
    ValueVec func_outputs;
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
    return Function(inputs, func_outputs, locals, globals, instructions);
}

Value FunctionBuilder::input(int index) const {
    if (index < 0 || index >= inputs.size()) {
        throw std::out_of_range(std::format(
            "Input index expected to be in range 0 to {}, got {}",
            inputs.size() - 1, index
        ));
    }
    return inputs.at(index);
}

ValueVec FunctionBuilder::input_range(int start_index, int end_index) const {
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
    return ValueVec(inputs.begin() + start_index, inputs.begin() + end_index);
}

void FunctionBuilder::output(int index, Value value) {
    if (index < 0 || index >= outputs.size()) {
        throw std::out_of_range(std::format(
            "Output index expected to be in range 0 to {}, got {}",
            outputs.size() - 1, index
        ));
    }
    auto& out_type = output_types.at(index);
    if (out_type.dtype != value.type.dtype || out_type.shape != value.type.shape) {
        throw std::invalid_argument(std::format("Wrong output type for output {}", index));
    }
    outputs.at(index) = value;
}

void FunctionBuilder::output_range(int start_index, const ValueVec& values) {
    if (start_index < 0 || start_index > outputs.size() - values.size()) {
        throw std::out_of_range(std::format(
            "Start index expected to be in range 0 to {}, got {}",
            outputs.size() - values.size(), start_index
        ));
    }
    std::copy(values.begin(), values.end(), outputs.begin() + start_index);
}

Value FunctionBuilder::global(
    const std::string& name, DataType dtype, const std::vector<int>& shape
) {
    Type type(dtype, BatchSize::one, shape);

    if (auto search = globals.find(name); search != globals.end()) {
        auto& found_global = search->second;
        if (type != found_global.type) {
            throw std::invalid_argument(std::format(
                "Conflicting types for global {}", name
            ));
        }
        return found_global;
    }

    int local_index = locals.size();
    Value new_global(type, local_index);
    locals.push_back(new_global);
    globals[name] = new_global;
    return new_global;
}

Value FunctionBuilder::sum(const ValueVec& values) {
    if (values.size() == 0) {
        return 0.0;
    }
    auto result = values.at(0);
    for (auto value = values.begin() + 1; value != values.end(); ++value) {
        result = add(result, *value);
    }
    return result;
}

/*Value FunctionBuilder::product(const ValueVec& values) {
    if (values.size() == 0) {
        return 1.;
    }
    auto result = values.at(0);
    for (auto value = values.begin() + 1; value != values.end(); ++value) {
        result = mul(result, *value);
    }
    return result;
}*/
