#include "madevent/runtime/runtime_base.h"

#include <dlfcn.h>
#include <format>

using namespace madevent;

namespace {

struct LoadedRuntime {
    inline static std::string lib_path = "";

    LoadedRuntime(const std::string& file) {
#ifdef __APPLE__
        std::string so_ext = "dylib";
#else
        std::string so_ext = "so";
#endif
        shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
            dlopen(std::format("{}/{}.{}", lib_path, file, so_ext).c_str(), RTLD_NOW),
            [](void* lib) { dlclose(lib); }
        );
        if (!shared_lib) {
            throw std::runtime_error(std::format(
                "Could not load shared object {}", file
            ));
        }
        get_device = reinterpret_cast<decltype(get_device)>(
            dlsym(shared_lib.get(), "get_device")
        );
        if (get_device == nullptr) {
            throw std::runtime_error(std::format(
                "Did not find symbol get_device in shared object {}", file
            ));
        }
        build_runtime = reinterpret_cast<decltype(build_runtime)>(
            dlsym(shared_lib.get(), "build_runtime")
        );
        if (build_runtime == nullptr) {
            throw std::runtime_error(std::format(
                "Did not find symbol build_runtime in shared object {}", file
            ));
        }
    }

    std::unique_ptr<void, std::function<void(void*)>> shared_lib;
    DevicePtr (*get_device)();
    Runtime* (*build_runtime)(const Function& function, ContextPtr context, bool concurrent);
};

const LoadedRuntime& cpu_runtime() {
    static LoadedRuntime runtime("libmadevent_cpu");
    return runtime;
}

const LoadedRuntime& cuda_runtime() {
    static LoadedRuntime runtime("libmadevent_cuda");
    return runtime;
}

}

RuntimePtr madevent::build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    if (context->device() == cpu_device()) {
        return RuntimePtr(cpu_runtime().build_runtime(function, context, concurrent));
    } else if (context->device() == cuda_device()) {
        return RuntimePtr(cuda_runtime().build_runtime(function, context, concurrent));
    } else {
        throw std::runtime_error("Invalid device");
    }
}

DevicePtr madevent::cpu_device() {
    return cpu_runtime().get_device();
}

DevicePtr madevent::cuda_device() {
    return cuda_runtime().get_device();
}

void madevent::set_lib_path(const std::string& lib_path) {
    LoadedRuntime::lib_path = lib_path;
}
