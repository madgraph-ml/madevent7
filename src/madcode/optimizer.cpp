#include "madevent/madcode/optimizer.h"

#include <algorithm>

using namespace madevent;

InstructionDependencies::InstructionDependencies(const Function& function) :
    size(function.instructions.size()), matrix(size * size)
{
    std::vector<int> local_source(function.locals.size(), -1);
    int index = 0;
    for (auto& instr : function.instructions) {
        for (auto& input : instr.inputs) {
            auto source_index = local_source[input.local_index];
            if (source_index != -1) {
                matrix[index * size + source_index] = true;
                for (int i = 0; i < size; ++i) {
                    matrix[index * size + i] =
                        matrix[index * size + i] | matrix[source_index * size + i];
                }
            }
        }
        for (auto& output : instr.outputs) {
            local_source[output.local_index] = index;
        }
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
