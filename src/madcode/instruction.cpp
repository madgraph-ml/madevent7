#include "madevent/madcode/instruction.h"

#include <map>
#include <algorithm>
#include <format>
#include <tuple>
#include <ranges>

using namespace madevent;

namespace {

std::tuple<int, std::string> get_offset_and_name(const std::string& shape_str) {
    int offset = std::ranges::count(shape_str, '+') - std::ranges::count(shape_str, '-');
    auto var_name = shape_str
                  | std::views::filter([](char c) { return c != '+' && c != '-'; })
                  | std::ranges::to<std::string>();
    return {offset, var_name};
}

}

TypeList SimpleInstruction::signature(
    const TypeList& args
) const {
    if (inputs.size() != args.size()) {
        throw std::invalid_argument(std::format(
            "Expected {} arguments, got {}", inputs.size(), args.size()
        ));
    }
    std::map<std::string, int> variables;
    std::vector<int> wildcard_shape;
    bool found_wildcard(false);
    BatchSize batch_size = BatchSize::one;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& [arg_dtype, arg_batch_size, arg_shape, _] = args.at(i);
        auto& [input_dtype, is_single, input_shape] = inputs.at(i);

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
                const auto shape_str = std::get_if<std::string>(&input_item);
                return shape_str && *shape_str == "...";
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
            auto insert_pos = mod_input_shape.erase(mod_input_shape.begin() + wildcard_index);
            mod_input_shape.insert(insert_pos, wildcard_shape.begin(), wildcard_shape.end());
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
                auto [offset, var_name] = get_offset_and_name(std::get<std::string>(input_item));
                if (auto var = variables.find(var_name); var != variables.end()) {
                    if (arg_item - offset != var->second) {
                        throw std::invalid_argument(std::format(
                            "{}, argument {}, dimension {}: expected size {}, got {}",
                            name, i + 1, j, var->second, arg_item
                        ));
                    }
                } else {
                    variables[var_name] = arg_item - offset;
                }
            }
        }
    }

    TypeList output_types;
    for (auto& [out_dtype, _, out_dyn_shape] : outputs) {
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
                    auto [offset, var_name] = get_offset_and_name(shape_str);
                    out_shape.push_back(variables.at(var_name) + offset);
                }
            }
        }
        output_types.push_back(Type{out_dtype, batch_size, out_shape});
    }
    return output_types;
}

TypeList StackInstruction::signature(const TypeList& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument("stack has to be called with at least one argument");
    }
    auto type = args.at(0);
    BatchSize batch_size = BatchSize::one;
    std::size_t i = 1;
    for (auto& arg : args) {
        if (arg.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(std::format(
                "stack, argument {}: Batch size list not accepted as argument", i
            ));
        }
        if (batch_size == BatchSize::one) {
            batch_size = arg.batch_size;
        } else if (arg.batch_size != BatchSize::one && batch_size != arg.batch_size) {
            throw std::invalid_argument(std::format(
                "{}, argument {}: incompatible batch size", name, i
            ));
        }
        if (arg != type) {
            throw std::invalid_argument("stack: all arguments must have the same shape and dtype");
        }
        ++i;
    }
    int args_size = args.size();
    std::vector<int> out_shape{args_size};
    out_shape.insert(out_shape.end(), type.shape.begin(), type.shape.end());
    return {{type.dtype, batch_size, out_shape}};
}

TypeList UnstackInstruction::signature(const TypeList& args) const {
    if (args.size() != 1) {
        throw std::invalid_argument(std::format(
            "unstack expects one argument, got {}", args.size()
        ));
    }
    auto arg = args.at(0);
    if (arg.dtype == DataType::batch_sizes) {
        throw std::invalid_argument("Batch size list not accepted as argument");
    }
    if (arg.batch_size == BatchSize::one) {
        throw std::invalid_argument("Argument must have batch dimension");
    }
    if (arg.shape.size() == 0) {
        throw std::invalid_argument("Argument of unstack must be at least one-dimensional");
    }
    std::vector<int> out_shape(arg.shape.begin() + 1, arg.shape.end());
    return TypeList(arg.shape[0], {arg.dtype, arg.batch_size, out_shape});
}

TypeList BatchCatInstruction::signature(const TypeList& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument("batch_cat has to be called with at least one argument");
    }
    auto type = args.at(0);
    auto batch_size = BatchSize::zero;
    std::vector<BatchSize> arg_batch_sizes;
    for (auto& arg : args) {
        if (arg.dtype == DataType::batch_sizes) {
            throw std::invalid_argument("Batch size list not accepted as argument");
        }
        if (arg.batch_size == BatchSize::one) {
            throw std::invalid_argument("Argument must have batch dimension");
        }
        if (arg.dtype != type.dtype || arg.shape != type.shape) {
            throw std::invalid_argument("All arguments must have the same shape and dtype");
        }
        arg_batch_sizes.push_back(arg.batch_size);
        batch_size = batch_size + arg.batch_size;
    }
    int args_size = args.size();
    return {{type.dtype, batch_size, type.shape}, arg_batch_sizes};
}

TypeList BatchSplitInstruction::signature(const TypeList& args) const {
    if (args.size() != 2) {
        throw std::invalid_argument(std::format(
            "batch_split expects two arguments, got {}", args.size()
        ));
    }
    auto split_arg = args.at(0);
    if (split_arg.batch_size == BatchSize::one) {
        throw std::invalid_argument("First argument of batch_split must have batch dimension");
    }
    auto count_arg = args.at(1);
    if (count_arg.dtype != DataType::batch_sizes) {
        throw std::invalid_argument("Second argument of batch_split must be batch size list");
    }
    TypeList out_types;
    auto last_batch_size = split_arg.batch_size;
    for (auto& batch_size : count_arg.batch_size_list) {
        if (&batch_size == &count_arg.batch_size_list.back()) {
            out_types.push_back({split_arg.dtype, last_batch_size, split_arg.shape});
        } else {
            out_types.push_back({split_arg.dtype, batch_size, split_arg.shape});
            last_batch_size = last_batch_size - batch_size;
        }
    }
    return out_types;
}

const std::unordered_map<std::string, InstructionOwner> madevent::build_instruction_set() {
#include "instruction_set_mixin.h"
    std::unordered_map<std::string, InstructionOwner> instruction_set;
    for (auto& instruction : instructions) {
        instruction_set[instruction->name] = std::move(instruction);
    }
    return instruction_set;
}
