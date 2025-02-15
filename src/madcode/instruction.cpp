#include "madevent/madcode/instruction.h"

#include <algorithm>
#include <format>
#include <tuple>
#include <ranges>

using namespace madevent;

ShapeExpr::ShapeExpr(const char* expr) {
    int state = 0;
    int sign = 1;
    int factor = 0;
    bool no_digits = true;
    for (int i = 0; expr[i] != '\0'; ++i) {
        char c = expr[i];
        int digit = -1;
        int sign_value = 0;
        bool is_var = false;
        if (c == ' ') {
            continue;
        } else if (c == '-') {
            sign_value = -1;
        } else if (c == '+') {
            sign_value = 1;
        } else if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'z') {
            is_var = true;
        } else {
            throw std::invalid_argument(std::format(
                "Invalid character {} in size expression", c
            ));
        }
        switch (state) {
        case 0: // first character
            if (sign_value != 0) {
                sign = sign_value;
                state = 1;
            } else if (digit != -1) {
                factor = digit;
                no_digits = false;
                state = 1;
            } else if (is_var) {
                terms.emplace_back(c, 1);
                state = 2;
            }
            break;
        case 1: // expect sign, digit or variable
            if (sign_value != 0) {
                if (no_digits) {
                    sign *= sign_value;
                } else {
                    terms.emplace_back(0, sign * factor);
                    sign = 1;
                    factor = 0;
                    no_digits = true;
                    state = 2;
                }
            } else if (digit != -1) {
                factor = 10 * factor + digit;
                no_digits = false;
            } else if (is_var) {
                terms.emplace_back(c, no_digits ? sign : sign * factor);
                sign = 1;
                factor = 0;
                no_digits = true;
                state = 2;
            }
            break;
        case 2: // expect sign
            if (sign_value != 0) {
                sign *= sign_value;
                state = 1;
            } else {
                throw std::invalid_argument("Invalid size expression");
            }
            break;
        }
    }
    if (!no_digits) {
        terms.emplace_back(0, sign * factor);
    }
}

bool ShapeExpr::check_and_update(std::map<char, int>& variables, int value) const {
    char unknown_var = 0;
    int unknown_factor = 0;
    int offset = 0;
    for (auto& [var_name, factor] : terms) {
        if (var_name == 0) {
            offset += factor;
        } else if (auto search = variables.find(var_name); search != variables.end()) {
            offset += factor * search->second;
        } else if (unknown_var == 0) {
            unknown_var = var_name;
            unknown_factor = factor;
        } else {
            return false;
        }
    }
    if (unknown_var == 0) {
        return value == offset;
    }
    if ((value - offset) % unknown_factor != 0) {
        return false;
    }
    variables[unknown_var] = (value - offset) / unknown_factor;
    return true;
}

std::optional<int> ShapeExpr::evaluate(const std::map<char, int>& variables) const {
    int value = 0;
    for (auto& [var_name, factor] : terms) {
        if (var_name == 0) {
            value += factor;
        } else if (auto search = variables.find(var_name); search != variables.end()) {
            value += factor * search->second;
        } else {
            return std::nullopt;
        }
    }
    return value;
}

TypeList SimpleInstruction::signature(const ValueList& args) const {
    if (inputs.size() != args.size()) {
        throw std::invalid_argument(std::format(
            "Expected {} arguments, got {}", inputs.size(), args.size()
        ));
    }
    std::map<char, int> variables;
    std::vector<int> wildcard_shape;
    bool found_wildcard(false);
    BatchSize batch_size = BatchSize::one;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& arg = args.at(i);
        auto& [arg_dtype, arg_batch_size, arg_shape, _] = arg.type;
        auto& [input_dtype, is_single, input_shape, is_size] = inputs.at(i);

        if (is_size) {
            if (
                arg_dtype != DataType::dt_int ||
                arg_batch_size != BatchSize::one ||
                arg_shape.size() != 0 ||
                std::holds_alternative<long long>(arg.literal_value)
            ) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: expected integer constant", name, i + 1
                ));
            }
            char var_name = std::get<ShapeExpr>(input_shape.at(0)).first_var_name();
            if (variables.find(var_name) != variables.end()) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: size already defined", name, i + 1
                ));
            }
            variables[var_name] = std::get<long long>(arg.literal_value);
            continue;
        }

        if (arg_dtype == DataType::batch_sizes) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: batch size list not accepted as argument", name, i + 1
            ));
        }

        if (input_dtype != arg_dtype) {
            std::cout << input_dtype << " " << arg_dtype << "\n";
            throw std::invalid_argument(std::format(
                "{}, argument {}: dtypes not matching", name, i + 1
            ));
        }

        if (is_single) {
            if (arg_batch_size != BatchSize::one) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: cannot have batch dimension", name, i + 1
                ));
            }
        } else {
            if (batch_size == BatchSize::one) {
                batch_size = arg_batch_size;
            } else if (arg_batch_size != BatchSize::one && batch_size != arg_batch_size) {
                throw std::invalid_argument(std::format(
                    "{}, argument {}: incompatible batch size", name, i + 1
                ));
            }
        }

        auto wildcard_pos = std::find_if(
            input_shape.begin(),
            input_shape.end(),
            [](const auto &input_item) {
                return std::holds_alternative<std::monostate>(input_item);
            }
        );
        auto mod_input_shape = input_shape;
        if (wildcard_pos != input_shape.end()) {
            auto wildcard_index = wildcard_pos - input_shape.begin();
            if (!found_wildcard) {
                if (arg_shape.size() < input_shape.size() - 1) {
                    throw std::invalid_argument(std::format(
                        "{}, argument {}: expected dimension of at least {}, got {}",
                        name, i + 1, input_shape.size() - 1, arg_shape.size()
                    ));
                }
                auto begin_pos = arg_shape.begin() + wildcard_index;
                auto end_pos = arg_shape.end() - (input_shape.end() - wildcard_pos) + 1;
                wildcard_shape.insert(wildcard_shape.begin(), begin_pos, end_pos);
                found_wildcard = true;
            }
            auto insert_pos = mod_input_shape.erase(
                mod_input_shape.begin() + wildcard_index
            );
            mod_input_shape.insert(
                insert_pos, wildcard_shape.begin(), wildcard_shape.end()
            );
        }

        if (arg_shape.size() != mod_input_shape.size()) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: expected dimension {}, got {}",
                name, i + 1, mod_input_shape.size(), arg_shape.size()
            ));
        }
        for (size_t j = 0; j < arg_shape.size(); ++j) {
            auto input_item = mod_input_shape[j];
            auto arg_item = arg_shape[j];
            if (const auto shape_int = std::get_if<int>(&input_item)) {
                if (arg_item != *shape_int) {
                    throw std::invalid_argument(std::format(
                        "{}, argument {}, dimension {}: expected size {}, got {}",
                        name, i + 1, j, *shape_int, arg_item
                    ));
                }
            } else {
                if (!std::get<ShapeExpr>(input_item).check_and_update(variables, arg_item)) {
                    throw std::invalid_argument(std::format(
                        "{}, argument {}, dimension {}: incompatible size", name, i + 1, j
                    ));
                }
            }
        }
    }

    TypeList output_types;
    for (auto& [out_dtype, is_single, out_dyn_shape, is_size] : outputs) {
        std::vector<int> out_shape;
        for (auto& shape_item : out_dyn_shape) {
            if (const auto shape_int = std::get_if<int>(&shape_item)) {
                out_shape.push_back(*shape_int);
            } else {
                if (std::holds_alternative<std::monostate>(shape_item)) {
                    if (!found_wildcard) {
                        throw std::invalid_argument(
                            "Wildcard found in output signature, but not in input"
                        );
                    }
                    out_shape.insert(
                        out_shape.end(), wildcard_shape.begin(), wildcard_shape.end()
                    );
                } else {
                    auto value = std::get<ShapeExpr>(shape_item).evaluate(variables);
                    if (!value) {
                        throw std::invalid_argument("Output size could not be determined");
                    }
                    out_shape.push_back(*value);
                }
            }
        }
        output_types.push_back(Type{out_dtype, batch_size, out_shape});
    }
    return output_types;
}

TypeList StackInstruction::signature(const ValueList& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument("stack has to be called with at least one argument");
    }
    auto type = args.at(0).type;
    BatchSize batch_size = BatchSize::one;
    std::size_t i = 1;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(std::format(
                "stack, argument {}: Batch size list not accepted as argument", i
            ));
        }
        if (batch_size == BatchSize::one) {
            batch_size = arg.type.batch_size;
        } else if (
            arg.type.batch_size != BatchSize::one && batch_size != arg.type.batch_size
        ) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: incompatible batch size", name, i
            ));
        }
        if (arg.type != type) {
            throw std::invalid_argument(
                "stack: all arguments must have the same shape and dtype"
            );
        }
        ++i;
    }
    int args_size = args.size();
    std::vector<int> out_shape{args_size};
    out_shape.insert(out_shape.end(), type.shape.begin(), type.shape.end());
    return {{type.dtype, batch_size, out_shape}};
}

TypeList UnstackInstruction::signature(const ValueList& args) const {
    if (args.size() != 1) {
        throw std::invalid_argument(std::format(
            "unstack expects one argument, got {}", args.size()
        ));
    }
    auto arg = args.at(0);
    if (arg.type.dtype == DataType::batch_sizes) {
        throw std::invalid_argument("Batch size list not accepted as argument");
    }
    if (arg.type.batch_size == BatchSize::one) {
        throw std::invalid_argument("Argument must have batch dimension");
    }
    if (arg.type.shape.size() == 0) {
        throw std::invalid_argument("Argument of unstack must be at least one-dimensional");
    }
    std::vector<int> out_shape(arg.type.shape.begin() + 1, arg.type.shape.end());
    return TypeList(arg.type.shape[0], {arg.type.dtype, arg.type.batch_size, out_shape});
}

TypeList BatchCatInstruction::signature(const ValueList& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument("batch_cat has to be called with at least one argument");
    }
    auto type = args.at(0).type;
    auto batch_size = BatchSize::zero;
    std::vector<BatchSize> arg_batch_sizes;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument("Batch size list not accepted as argument");
        }
        if (arg.type.batch_size == BatchSize::one) {
            throw std::invalid_argument("Argument must have batch dimension");
        }
        if (arg.type.dtype != type.dtype || arg.type.shape != type.shape) {
            throw std::invalid_argument("All arguments must have the same shape and dtype");
        }
        arg_batch_sizes.push_back(arg.type.batch_size);
        batch_size = batch_size + arg.type.batch_size;
    }
    int args_size = args.size();
    return {{type.dtype, batch_size, type.shape}, arg_batch_sizes};
}

TypeList BatchSplitInstruction::signature(const ValueList& args) const {
    if (args.size() != 2) {
        throw std::invalid_argument(std::format(
            "batch_split expects two arguments, got {}", args.size()
        ));
    }
    auto split_arg = args.at(0);
    if (split_arg.type.batch_size == BatchSize::one) {
        throw std::invalid_argument(
            "First argument of batch_split must have batch dimension"
        );
    }
    auto count_arg = args.at(1);
    if (count_arg.type.dtype != DataType::batch_sizes) {
        throw std::invalid_argument(
            "Second argument of batch_split must be batch size list"
        );
    }
    TypeList out_types;
    auto last_batch_size = split_arg.type.batch_size;
    for (auto& batch_size : count_arg.type.batch_size_list) {
        if (&batch_size == &count_arg.type.batch_size_list.back()) {
            out_types.push_back({
                split_arg.type.dtype, last_batch_size, split_arg.type.shape
            });
        } else {
            out_types.push_back({
                split_arg.type.dtype, batch_size, split_arg.type.shape
            });
            last_batch_size = last_batch_size - batch_size;
        }
    }
    return out_types;
}

TypeList RqsActivationInstruction::signature(const ValueList& args) const {
    return {}; //TODO: implement
}

const std::unordered_map<std::string, InstructionOwner> madevent::build_instruction_set() {
#include "instruction_set_mixin.h"
    std::unordered_map<std::string, InstructionOwner> instruction_set;
    for (auto& instruction : instructions) {
        instruction_set[instruction->name] = std::move(instruction);
    }
    return instruction_set;
}
