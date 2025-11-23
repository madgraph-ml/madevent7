#include "madevent/runtime/context.h"

#include <dlfcn.h>

using namespace madevent;

MatrixElementApi::MatrixElementApi(
    const std::string& file, const std::string& param_card
) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(
            std::format("Could not load shared object {}\n{}", file, dlerror())
        );
    }

    SubProcessInfo* (*subprocess_info)() = reinterpret_cast<decltype(subprocess_info)>(
        dlsym(_shared_lib.get(), "subprocess_info")
    );
    if (subprocess_info == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol subprocess_info in shared object {}", file)
        );
    }
    _subprocess_info = *subprocess_info();

    _init_subprocess = reinterpret_cast<decltype(_init_subprocess)>(
        dlsym(_shared_lib.get(), "init_subprocess")
    );
    if (_init_subprocess == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol init_subprocess in shared object {}", file)
        );
    }

    _compute_matrix_element = reinterpret_cast<decltype(_compute_matrix_element)>(
        dlsym(_shared_lib.get(), "compute_matrix_element")
    );
    if (_compute_matrix_element == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol compute_matrix_element in shared object {}", file
            )
        );
    }

    _compute_matrix_element_multichannel =
        reinterpret_cast<decltype(_compute_matrix_element_multichannel)>(
            dlsym(_shared_lib.get(), "compute_matrix_element_multichannel")
        );
    if (_compute_matrix_element_multichannel == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol compute_matrix_element_multichannel in shared "
                "object {}",
                file
            )
        );
    }

    _free_subprocess = reinterpret_cast<decltype(_free_subprocess)>(
        dlsym(_shared_lib.get(), "free_subprocess")
    );
    if (_free_subprocess == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol free_subprocess in shared object {}", file)
        );
    }

    _instances = ThreadResource<InstanceType>(default_thread_pool(), [&] {
        return InstanceType(_init_subprocess(param_card.c_str()), [this](void* proc) {
            _free_subprocess(proc);
        });
    });
}

std::size_t
Context::load_matrix_element(const std::string& file, const std::string& param_card) {
    matrix_elements.emplace_back(file, param_card);
    return matrix_elements.size() - 1;
}

Tensor Context::define_global(
    const std::string& name, DataType dtype, const SizeVec& shape, bool requires_grad
) {
    SizeVec full_shape{1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (globals.contains(name)) {
        throw std::invalid_argument(
            std::format("Context already contains a global named {}", name)
        );
    }
    Tensor tensor(dtype, full_shape, _device);
    tensor.zero();
    globals[name] = {tensor, requires_grad};
    return tensor;
}

Tensor Context::global(const std::string& name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return std::get<0>(search->second);
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

bool Context::global_exists(const std::string& name) {
    return globals.find(name) != globals.end();
}

bool Context::global_requires_grad(const std::string& name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return std::get<1>(search->second);
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

const MatrixElementApi& Context::matrix_element(std::size_t index) const {
    if (index >= matrix_elements.size()) {
        throw std::runtime_error("Matrix element index out of bounds");
    }
    return matrix_elements[index];
}

void Context::save(const std::string& file) const {}

void Context::load(const std::string& file) {}

ContextPtr madevent::default_context() {
    static ContextPtr context = std::make_shared<Context>(cpu_device());
    return context;
}

ContextPtr madevent::default_cuda_context() {
    static ContextPtr context = std::make_shared<Context>(cuda_device());
    return context;
}

ContextPtr madevent::default_hip_context() {
    static ContextPtr context = std::make_shared<Context>(hip_device());
    return context;
}
