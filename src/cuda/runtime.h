#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/madcode/function.h"

#include <memory>

namespace madevent {
namespace cuda {

class Runtime {
public:
    Runtime(const Function& function);
    ~Runtime();
    std::vector<Tensor> run(std::vector<Tensor>& inputs) const;

private:
    // Hide implementation as it contains a few CUDA-specific types
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}
}
