#pragma once

#include <unordered_map>
#include <stdint.h>

#include "madevent/madcode.h"
#include "madevent/runtime/tensor.h"

namespace madevent {

struct SubProcessInfo {
    uint8_t on_gpu;
    uint64_t particle_count;
    uint64_t diagram_count;
    uint64_t amplitude_count;
    uint64_t helicity_count;
};

class MatrixElement {
public:
    MatrixElement(const std::string& file, const std::string& param_card);
    MatrixElement(MatrixElement&&) noexcept = default;
    MatrixElement& operator=(MatrixElement&&) noexcept = default;
    MatrixElement(const MatrixElement&) = delete;
    MatrixElement& operator=(const MatrixElement&) = delete;
    //~MatrixElement();
    void call(
        Tensor momenta_in, Tensor flavor_in, Tensor mirror_in, Tensor matrix_element_out
    ) const;
    void call_multichannel(
        Tensor momenta_in,
        Tensor alpha_s_in,
        Tensor random_in,
        Tensor flavor_in,
        Tensor mirror_in,
        Tensor amp2_remap_in,
        Tensor matrix_element_out,
        Tensor channel_weights_out,
        Tensor color_out,
        Tensor diagram_out
    ) const;
    bool on_gpu() const { return _subprocess_info.on_gpu; }
    std::size_t particle_count() const { return _subprocess_info.particle_count; }
    std::size_t diagram_count() const { return _subprocess_info.diagram_count; }
    std::size_t amplitude_count() const { return _subprocess_info.amplitude_count; }
    std::size_t helicity_count() const { return _subprocess_info.helicity_count; }

private:
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    SubProcessInfo _subprocess_info;
    void* (*_init_subprocess)(const char*);
    void (*_compute_matrix_element)(
        void*, uint64_t, uint64_t, const double*, const int64_t*, const int64_t*, double*
    );
    void (*_compute_matrix_element_multichannel)(
        void*, uint64_t, uint64_t, uint64_t, const double*, const double*, const double*,
        const int64_t*, const int64_t*, const int64_t*, double*, double*, int64_t*, int64_t*
    );
    void (*_free_subprocess)(void*);
    std::vector<std::unique_ptr<void, std::function<void(void*)>>> _process_instances;
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
    const MatrixElement& matrix_element(std::size_t index) const;
    void save(const std::string& file) const;
    void load(const std::string& file);
    DevicePtr device() { return _device; }

private:
    DevicePtr _device;
    std::unordered_map<std::string, std::tuple<Tensor, bool>> globals;
    std::vector<MatrixElement> matrix_elements;
};

using ContextPtr = std::shared_ptr<Context>;

ContextPtr default_context();
ContextPtr default_cuda_context();

inline std::string prefixed_name(const std::string& prefix, const std::string& name) {
    return prefix == "" ? name : std::format("{}.{}", prefix, name);
}

}
