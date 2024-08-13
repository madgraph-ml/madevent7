#include "madevent/madcode/runtime.h"

#include "kernels.h"

using namespace madevent::cpu;

static void parallel_for(size_t count, std::function<void(size_t)> function) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        function(i);
    }
}

template<class... T>
static size_t check_batch_dim(T... dims) {

}

template<F, class... T>
static void batch_foreach(F function, T... args) {
    auto batch_dim = check_batch_dim(args.shape[0]...);
    parallel_for(batch_dim, [](size_t i) {
        Accessor(args.data, args.shape[0])...);
    }

}

std::vector<Tensor> Runtime::run(std::vector<Tensor>& inputs) const {

}
