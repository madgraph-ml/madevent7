#pragma once

#include "madevent/madcode/function.h"

#include <vector>

namespace madevent {

class InstructionDependencies {
public:
    InstructionDependencies(const Function& function);
    bool depends(std::size_t test_index, std::size_t dependency_index) {
        return _matrix[test_index * _size + dependency_index];
    }
    const std::vector<int>& ranks() const { return _ranks; }

private:
    std::size_t _size;
    std::vector<bool> _matrix;
    std::vector<int> _ranks;
};


class LastUseOfLocals {
public:
    LastUseOfLocals(const Function& function);
    std::vector<int>& local_indices(std::size_t index) {
        return _last_used[index];
    }

private:
    std::vector<std::vector<int>> _last_used;
};

}
