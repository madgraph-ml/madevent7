#pragma once

#include "madevent/madcode/function.h"

#include <vector>

namespace madevent {

class InstructionDependencies {
public:
    InstructionDependencies(const Function& function);
    bool depends(std::size_t test_index, std::size_t dependency_index) {
        return matrix[test_index * size + dependency_index];
    }

private:
    friend class MergeOptimizer;
    std::size_t size;
    std::vector<bool> matrix;
    std::vector<int> ranks;
};


class LastUseOfLocals {
public:
    LastUseOfLocals(const Function& function);
    std::vector<int>& local_indices(std::size_t index) {
        return last_used[index];
    }

private:
    std::vector<std::vector<int>> last_used;
};

/*class MergeOptimizer {
public:
    MergeOptimizer(const Function& _function);
    Function optimize();
private:
    struct MergedInstruction {
        std::vector<InstructionCall> instructions;
        int rank;
        std::vector<bool> dependencies;
        bool active = true;
    };
    std::vector<MergedInstruction> instructions;
    const Function& function;

    void merge_if_compatible(
        std::size_t idx1, MergedInstruction& instr1, std::size_t idx2, MergedInstruction& instr2
    );
    Function build_function();
};*/

}
