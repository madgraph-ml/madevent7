#include "madevent/madcode/optimizer.h"

#include <algorithm>
#include <numeric>
#include <ranges>

using namespace madevent;

namespace {

std::vector<int> find_permutation(std::vector<int>& from, std::vector<int>& to) {
    auto indices = std::views::iota(from.size());
    std::unordered_map<int, int> from_location(std::from_range, std::views::zip(from, indices));
    return indices | std::views::transform([&](int i) { return from_location[to[i]]; })
                   | std::ranges::to<std::vector<int>>();
}

}

Function madevent::optimize_constants(const Function& function) {
    // add, sub, mul, clip_min, sqrt, square
    FunctionBuilder fb(function);
    ValueList new_locals(function.locals);
    for (auto& instr : function.instructions) {
        bool const_opt = true;
        ValueList inputs;
        for (auto& input : instr.inputs) {
            Value new_input = std::holds_alternative<std::monostate>(input.literal_value) ?
                new_locals.at(input.local_index) : Value(input.type, input.literal_value);
            inputs.push_back(new_input);
            if (std::holds_alternative<std::monostate>(new_input.literal_value)) {
                const_opt = false;
            }
        }
        double result;
        if (const_opt) {
            switch (instr.instruction->opcode) {
            case Opcode::add: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                double arg1 = std::get<double>(inputs.at(1).literal_value);
                result = arg0 + arg1;
                break;
            } case Opcode::sub: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                double arg1 = std::get<double>(inputs.at(1).literal_value);
                result = arg0 - arg1;
                break;
            } case Opcode::mul: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                double arg1 = std::get<double>(inputs.at(1).literal_value);
                result = arg0 * arg1;
                break;
            } case Opcode::clip_min: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                double arg1 = std::get<double>(inputs.at(1).literal_value);
                result = arg0 < arg1 ? arg1 : arg0;
                break;
            } case Opcode::sqrt: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                result = std::sqrt(arg0);
                break;
            } case Opcode::square: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                result = arg0 * arg0;
                break;
            } case Opcode::pow: {
                double arg0 = std::get<double>(inputs.at(0).literal_value);
                double arg1 = std::get<double>(inputs.at(1).literal_value);
                result = std::pow(arg0, arg1);
                break;
            } default:
                const_opt = false;
            }
        }

        if (const_opt) {
            new_locals.at(instr.outputs.at(0).local_index) = result;
        } else {
            auto outputs = fb.instruction(instr.instruction, inputs);
            for (auto [instr_output, output] : std::views::zip(instr.outputs, outputs)) {
                new_locals.at(instr_output.local_index) = output;
            }
        }
    }

    ValueList outputs;
    for (auto& output : function.outputs) {
        outputs.push_back(new_locals.at(output.local_index));
    }
    fb.output_range(0, outputs);
    return fb.function();
}

InstructionDependencies::InstructionDependencies(const Function& function) :
    size(function.instructions.size()), matrix(size * size)
{
    std::vector<int> local_source(function.locals.size(), -1);
    int index = 0;
    for (auto& instr : function.instructions) {
        int rank = 0;
        for (auto& input : instr.inputs) {
            auto source_index = local_source[input.local_index];
            if (source_index == -1) continue;
            matrix[index * size + source_index] = true;
            for (int i = 0; i < size; ++i) {
                matrix[index * size + i] =
                    matrix[index * size + i] | matrix[source_index * size + i];
            }
            int source_rank = ranks[source_index];
            if (rank < source_rank) rank = source_rank;
        }
        for (auto& output : instr.outputs) {
            local_source[output.local_index] = index;
        }
        ranks.push_back(rank + 1);
        ++index;
    }
}

LastUseOfLocals::LastUseOfLocals(const Function& function) :
    last_used(function.instructions.size())
{
    std::vector<bool> seen_locals;
    for (auto& local : function.locals) {
        seen_locals.push_back(!std::holds_alternative<std::monostate>(local.literal_value));
    }
    for (auto& output : function.outputs) {
        seen_locals[output.local_index] = true;
    }
    auto instr = function.instructions.rbegin();
    auto indices = last_used.begin();
    for (; instr != function.instructions.rend(); ++instr, ++indices) {
        for (auto& input : instr->inputs) {
            auto index = input.local_index;
            if (!seen_locals[index]) {
                indices->push_back(index);
                seen_locals[index] = true;
            }
        }
    }
    std::reverse(last_used.begin(), last_used.end());
}

MergeOptimizer::MergeOptimizer(const Function& _function) : function(_function) {
    InstructionDependencies dependencies(function);
    auto size = function.instructions.size();
    std::vector<std::size_t> perm(size);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&dependencies](auto i, auto j) {
        return dependencies.ranks[i] < dependencies.ranks[j];
    });
    for (auto i : perm) {
        MergedInstruction instr{{function.instructions[i]}, dependencies.ranks[i]};
        for (auto j : perm) {
            instr.dependencies.push_back(dependencies.matrix[i * size + j]);
        }
        instructions.push_back(instr);
    }
}

Function MergeOptimizer::optimize() {
    int max_rank = instructions.back().rank;
    std:size_t instr_size = instructions.size();
    for (int rank_diff = 0; rank_diff < max_rank; ++rank_diff) {
        for (std::size_t idx1 = 0; idx1 < instr_size; ++idx1) {
            auto& instr1 = instructions[idx1];
            if (!instr1.active) continue;
            for (std::size_t idx2 = idx1 + 1; idx2 < instr_size; ++idx2) {
                auto& instr2 = instructions[idx2];
                if (instr2.rank - instr1.rank > rank_diff) break;
                if (!instr2.active) continue;
                merge_if_compatible(idx1, instr1, idx2, instr2);
            }
        }
    }

    int count = -1;
    for (auto& instr : instructions) {
        ++count;
        if (!instr.active) continue;
        if (instr.instructions.size() == 1) {
            std::cout << count << " [" << instr.rank << "] " << instr.instructions.front() << "\n";
        } else {
            std::cout << count << " [" << instr.rank << "] {\n";
            for (auto& sub_instr : instr.instructions) {
                std::cout << "  " << sub_instr << "\n";
            }
            std::cout << "}\n";
        }
    }
    return build_function();
}

Function MergeOptimizer::build_function() {
    struct SourceIndices {
        int instr;
        int output;
        int sub_instr;
    };
    FunctionBuilder fb(function);
    std::vector<SourceIndices> local_source_indices(function.locals.size());
    std::vector<std::vector<SourceIndices>> input_source_instr;
    std::vector<std::vector<int>> input_local_indices;
    std::vector<int> permutation, arg_source_locals, arg_locals;
    int instr_index = -1;
    for (auto& instr : instructions) {
        ++instr_index;
        if (!instr.active) continue;
        auto input_count = instr.instructions[0].inputs.size();
        input_source_instr.assign(input_count, {});
        input_local_indices.assign(input_count, {});
        int sub_instr_index = 0;
        for (auto& sub_instr : instr.instructions) {
            for (auto&& [input, isrc, iloc] : std::views::zip(
                sub_instr.inputs, input_source_instr, input_local_indices
            )) {
                isrc.push_back(local_source_indices[input.local_index]);
                iloc.push_back(input.local_index);
            }
            int output_index = 0;
            for (auto& output : sub_instr.outputs) {
                local_source_indices[output.local_index] = {
                    instr_index, output_index, sub_instr_index
                };
                ++output_index;
            }
            ++sub_instr_index;
        }

        permutation.clear();
        arg_source_locals.clear();
        arg_locals.clear();
        for (auto&& [isrc, iloc] : std::views::zip(input_source_instr, input_local_indices)) {
            if (std::adjacent_find(isrc.begin(), isrc.end(), [](auto& a, auto& b) {
                return a.instr != b.instr || a.output != b.output;
            }) == isrc.end()) {
                // all have the same source
                for (auto& source_instr : instructions[isrc[0].instr].instructions) {
                    arg_source_locals.push_back(source_instr.outputs[isrc[0].output].local_index);
                }

                if (permutation.size() > 0) {
                    auto next_permutation = find_permutation(iloc, arg_source_locals);
                    if (std::equal(permutation.begin(), permutation.end(), next_permutation.begin())) {

                    } else {
                        permutation = next_permutation;
                    }
                } else {
                    permutation = find_permutation(iloc, arg_source_locals);
                }

                // determine permutation
                // if permutation exists
                //     if permutation is the same
                //         use value before split
                //     else
                //         use cat instruction
                // else
                //     set permutation
                //     use value before split
            } else {
                // use cat instruction
            }
            // emit cat instructions with permutation
        }
        // emit instruction
        // for each output
        //     emit split instruction with permutation
    }
    return fb.function();
}

void MergeOptimizer::merge_if_compatible(
    std::size_t idx1, MergedInstruction& instr1, std::size_t idx2, MergedInstruction& instr2
) {
    //return;
    // Check if instructions can be merged, otherwise return
    auto& instr1_first = instr1.instructions.front();
    auto& instr2_first = instr2.instructions.front();
    if (
        instr1.dependencies[idx2] ||
        instr2.dependencies[idx1] ||
        instr1_first.instruction != instr2_first.instruction ||
        instr1_first.instruction->opcode == Opcode::batch_cat ||
        instr1_first.instruction->opcode == Opcode::batch_split ||
        instr1_first.inputs.size() != instr2_first.inputs.size()
    ) return;
    for (
        auto in1 = instr1_first.inputs.begin(), in2 = instr2_first.inputs.begin();
        in1 != instr1_first.inputs.end();
        ++in1, ++in2
    ) {
        if (in1->literal_value != in2->literal_value) return;
        if (!std::holds_alternative<std::monostate>(in1->literal_value)) continue;
        if (in1->type != in2->type) return;
    }

    // Deactivate first instruction, move content to second one
    instr1.active = false;
    instr2.instructions.insert(
        instr2.instructions.end(),
        std::make_move_iterator(instr1.instructions.begin()),
        std::make_move_iterator(instr1.instructions.end())
    );
    instr1.instructions.clear();
    auto dep1 = instr1.dependencies, dep2 = instr2.dependencies;

    // Permute instructions and dependency matrix such that instructions which depended on the first
    // instruction are executed after the second one
    std::vector<std::size_t> perm_indices;
    std::vector<std::size_t> indices_after;
    for (std::size_t i = idx1 + 1; i < idx2; ++i) {
        if (instructions[i].dependencies[idx1]) {
            indices_after.push_back(i);
        } else {
            perm_indices.push_back(i);
        }
    }
    auto new_idx2 = idx1 + 1 + perm_indices.size();
    perm_indices.push_back(idx2);
    perm_indices.insert(perm_indices.end(), indices_after.begin(), indices_after.end());
    std::vector<MergedInstruction> perm_instructions;
    for (auto perm_index : perm_indices) {
        perm_instructions.push_back(std::move(instructions[perm_index]));
    }
    std::move(perm_instructions.begin(), perm_instructions.end(), instructions.begin() + idx1 + 1);
    std::vector<bool> perm_dep(idx2 - idx1);
    for (auto instr = instructions.begin() + idx1; instr != instructions.end(); ++instr) {
        perm_dep.clear();
        for (auto perm_index : perm_indices) {
            perm_dep.push_back(instr->dependencies[perm_index]);
        }
        std::copy(perm_dep.begin(), perm_dep.end(), instr->dependencies.begin() + idx1 + 1);
    }

    // Update dependency matrix to account for the merge
    auto& new_instr2 = instructions[new_idx2];
    dep2[new_idx2] = true;
    instr1.dependencies.assign(dep1.size(), false); // TODO: is this needed?
    std::transform(
        new_instr2.dependencies.begin(), new_instr2.dependencies.end(), dep1.begin(),
        new_instr2.dependencies.begin(), std::logical_or<>{}
    );
    for (auto instr = instructions.begin() + new_idx2 + 1; instr != instructions.end(); ++instr) {
        auto& dep = instr->dependencies;
        if (dep[idx1]) {
            std::transform(dep.begin(), dep.end(), dep2.begin(), dep.begin(), std::logical_or<>{});
            dep[idx1] = false;
        }
        if (dep[new_idx2]) {
            std::transform(dep.begin(), dep.end(), dep1.begin(), dep.begin(), std::logical_or<>{});
        }
    }
}
