#include "madevent/madcode/instruction.h"

#include <map>
#include <algorithm>
#include <fmt/ranges.h>

using namespace madevent;

const TypeList SimpleInstruction::signature(
    const TypeList& args
) const {
    if (inputs.size() != args.size()) {
        throw std::invalid_argument(fmt::format(
            "Expected {} arguments, got {}", inputs.size(), args.size()
        ));
    }
    std::map<std::string, int> variables;
    std::vector<int> wildcard_shape;
    bool found_wildcard(false);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& [arg_dtype, arg_shape] = args[i];
        auto& [input_dtype, input_shape] = inputs[i];

        if (input_dtype != arg_dtype) {
            throw std::invalid_argument(fmt::format(
                "Argument {}: dtypes not matching", i + 1
            ));
        }

        auto wildcard_pos = std::find_if(
            input_shape.begin(),
            input_shape.end(),
            [](const auto &input_item) {
                const auto shape_str = std::get_if<std::string>(&input_item);
                return shape_str && *shape_str == "...";
            }
        );
        auto mod_input_shape = input_shape;
        if (wildcard_pos != input_shape.end()) {
            auto wildcard_index = wildcard_pos - input_shape.begin();
            if (!found_wildcard) {
                if (arg_shape.size() < input_shape.size() - 1) {
                    throw std::invalid_argument(fmt::format(
                        "Argument {}: expected dimension of at least {}, got {}",
                        i, input_shape.size() - 1, arg_shape.size()
                    ));
                }
                auto begin_pos = arg_shape.begin() + wildcard_index;
                auto end_pos = arg_shape.end() - (input_shape.end() - wildcard_pos) + 1;
                wildcard_shape.insert(wildcard_shape.begin(), begin_pos, end_pos);
                found_wildcard = true;
            }
            auto insert_pos = mod_input_shape.erase(mod_input_shape.begin() + wildcard_index);
            mod_input_shape.insert(insert_pos, wildcard_shape.begin(), wildcard_shape.end());
        }

        if (arg_shape.size() != mod_input_shape.size()) {
            throw std::invalid_argument(fmt::format(
                "Argument {}: expected dimension {}, got {}",
                i, mod_input_shape.size(), arg_shape.size()
            ));
        }
        for (size_t j = 0; j < arg_shape.size(); ++j) {
            auto input_item = mod_input_shape[j];
            auto arg_item = arg_shape[j];
            if (const auto shape_int = std::get_if<int>(&input_item)) {
                if (arg_item != *shape_int) {
                    throw std::invalid_argument(fmt::format(
                        "Argument {}, dimension {}: expected size {}, got {}",
                        i, j, *shape_int, arg_item
                    ));
                }
            } else {
                auto shape_str = std::get<std::string>(input_item);
                if (auto var = variables.find(shape_str); var != variables.end()) {
                    if (arg_item != var->second) {
                        throw std::invalid_argument(fmt::format(
                            "Argument {}, dimension {}: expected size {}, got {}",
                            i, j, var->second, arg_item
                        ));
                    }
                } else {
                    variables[shape_str] = arg_item;
                }
            }

        }
    }

    TypeList output_types;
    for (auto& [out_dtype, out_dyn_shape] : outputs) {
        std::vector<int> out_shape;
        for (auto& shape_item : out_dyn_shape) {
            if (const auto shape_int = std::get_if<int>(&shape_item)) {
                out_shape.push_back(*shape_int);
            } else {
                auto shape_str = std::get<std::string>(shape_item);
                if (shape_str == "...") {
                    if (!found_wildcard) {
                        throw std::invalid_argument(
                            "Wildcard found in output signature, but not in input"
                        );
                    }
                    out_shape.insert(
                        out_shape.end(), wildcard_shape.begin(), wildcard_shape.end()
                    );
                } else {
                    out_shape.push_back(variables.at(shape_str));
                }
            }
        }
        output_types.push_back(Type{out_dtype, out_shape});
    }
    return output_types;
}

const std::unordered_map<std::string, InstructionPtr> madevent::build_instruction_set() {
#include "instruction_set_mixin.h"
    std::unordered_map<std::string, InstructionPtr> instruction_set;
    for (auto& instruction : instructions) {
        instruction_set[instruction->name] = std::move(instruction);
    }
    return instruction_set;
}
