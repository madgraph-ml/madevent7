#include "madevent/runtime/context.h"

#include <dlfcn.h>
#include <unordered_map>

using namespace madevent;

MatrixElementApi::MatrixElementApi(
    const std::string& file, const std::string& param_card, std::size_t index
) :
    _file_name(file), _index(index) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(
            std::format("Could not load shared object {}\n{}", file, dlerror())
        );
    }

    _get_meta = reinterpret_cast<decltype(&umami_get_meta)>(
        dlsym(_shared_lib.get(), "umami_get_meta")
    );
    if (_get_meta == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_get_meta in shared object {}", file)
        );
    }

    _initialize = reinterpret_cast<decltype(&umami_initialize)>(
        dlsym(_shared_lib.get(), "umami_initialize")
    );
    if (_initialize == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_initialize in shared object {}", file
            )
        );
    }

    _matrix_element = reinterpret_cast<decltype(&umami_matrix_element)>(
        dlsym(_shared_lib.get(), "umami_matrix_element")
    );
    if (_matrix_element == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_matrix_element in shared object {}", file
            )
        );
    }

    _free =
        reinterpret_cast<decltype(&umami_free)>(dlsym(_shared_lib.get(), "umami_free"));
    if (_free == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_free in shared object {}", file)
        );
    }

    _instances = ThreadResource<InstanceType>(default_thread_pool(), [&] {
        void* instance;
        check_umami_status(_initialize(&instance, param_card.c_str()));
        return InstanceType(instance, [this](void* proc) { _free(proc); });
    });
}

void MatrixElementApi::check_umami_status(UmamiStatus status) const {
    std::string error;
    switch (status) {
    case UMAMI_SUCCESS:
        return;
    case UMAMI_ERROR:
        throw_error("unspecified error");
    case UMAMI_ERROR_NOT_IMPLEMENTED:
        throw_error("functionality not implemented");
    case UMAMI_ERROR_UNSUPPORTED_INPUT:
        throw_error("unsupported input key");
    case UMAMI_ERROR_UNSUPPORTED_OUTPUT:
        throw_error("unsupported output key");
    case UMAMI_ERROR_UNSUPPORTED_META:
        throw_error("unsupported metadata key");
    case UMAMI_ERROR_MISSING_INPUT:
        throw_error("missing input");
    default:
        throw_error("unknown error");
    }
}

void MatrixElementApi::throw_error(const std::string& message) const {
    throw std::runtime_error(
        std::format("Error in call to matrix element API {}: {}", _file_name, message)
    );
}

const MatrixElementApi&
Context::load_matrix_element(const std::string& file, const std::string& param_card) {
    return matrix_elements.emplace_back(file, param_card, matrix_elements.size());
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
    static ContextPtr context = default_device_context(cpu_device());
    return context;
}

ContextPtr madevent::default_cuda_context() {
    static ContextPtr context = default_device_context(cuda_device());
    return context;
}

ContextPtr madevent::default_hip_context() {
    static ContextPtr context = default_device_context(hip_device());
    return context;
}

ContextPtr madevent::default_device_context(DevicePtr device) {
    static std::unordered_map<DevicePtr, ContextPtr> default_contexts;
    if (auto search = default_contexts.find(device); search != default_contexts.end()) {
        return search->second;
    } else {
        ContextPtr context = std::make_shared<Context>(device);
        default_contexts[device] = context;
        return context;
    }
}
