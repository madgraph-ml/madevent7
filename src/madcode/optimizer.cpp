#include "madevent/madcode/optimizer.h"

using namespace madevent;

InstructionDependencies::InstructionDependencies(Function& function) :
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
                    matrix[index * size + i] |= matrix[source_index * size + i];
                }
            }
        }
        for (auto& output : instr.outputs) {
            local_source[output.local_index] = index;
        }
        ++index;
    }
}
