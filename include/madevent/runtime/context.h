#pragma once

#include <unordered_map>
#include <stdint.h>

#include "madevent/madcode.h"
#include "madevent/runtime/tensor.h"
#include "madevent/runtime/thread_pool.h"

namespace madevent {

struct SubProcessInfo {
    uint8_t on_gpu;
    uint64_t particle_count;
    uint64_t diagram_count;
    uint64_t helicity_count;
};

class MatrixElementApi {
public:
    MatrixElementApi(const std::string& file, const std::string& param_card);
    MatrixElementApi(MatrixElementApi&&) noexcept = default;
    MatrixElementApi& operator=(MatrixElementApi&&) noexcept = default;
    MatrixElementApi(const MatrixElementApi&) = delete;
    MatrixElementApi& operator=(const MatrixElementApi&) = delete;
    bool on_gpu() const { return _subprocess_info.on_gpu; }
    std::size_t particle_count() const { return _subprocess_info.particle_count; }
    std::size_t diagram_count() const { return _subprocess_info.diagram_count; }
    std::size_t helicity_count() const { return _subprocess_info.helicity_count; }

    template<typename... T>
    void call(T... args) const { _compute_matrix_element(std::forward<T>(args)...); }
    template<typename... T>
    void call_multichannel(T... args) const {
        _compute_matrix_element_multichannel(std::forward<T>(args)...);
    }
    void* process_instance(std::size_t index) const {
        return _instances.get(index).get();
    }

private:
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    SubProcessInfo _subprocess_info;
    void* (*_init_subprocess)(const char*);
    void (*_compute_matrix_element)(
        void*, uint64_t, uint64_t, const double*, const int64_t*, const int64_t*, double*
    );
    void (*_compute_matrix_element_multichannel)(
        void*, uint64_t, uint64_t, const double*, const double*, const double*,
        const int64_t*, const int64_t*, double*, double*, int64_t*, int64_t*, int64_t*
    );
    void (*_free_subprocess)(void*);
    using InstanceType = std::unique_ptr<void, std::function<void(void*)>>;
    ThreadResource<InstanceType> _instances;
};

class Context {
    /**
     * Contains global variables and matrix elements
     */
public:
    Context() : _device(cpu_device()) {}
    Context(DevicePtr device) : _device(device) {}
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    std::size_t load_matrix_element(
        const std::string& file, const std::string& param_card
    );
    Tensor define_global(
        const std::string& name,
        DataType dtype,
        const SizeVec& shape,
        bool requires_grad=false
    );
    Tensor global(const std::string& name);
    bool global_requires_grad(const std::string& name);
    bool global_exists(const std::string& name);
    const MatrixElementApi& matrix_element(std::size_t index) const;
    void save(const std::string& file) const;
    void load(const std::string& file);
    DevicePtr device() { return _device; }

private:
    DevicePtr _device;
    std::unordered_map<std::string, std::tuple<Tensor, bool>> globals;
    std::vector<MatrixElementApi> matrix_elements;
};

using ContextPtr = std::shared_ptr<Context>;

ContextPtr default_context();
ContextPtr default_cuda_context();

inline std::string prefixed_name(const std::string& prefix, const std::string& name) {
    return prefix == "" ? name : std::format("{}.{}", prefix, name);
}

}
