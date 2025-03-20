#pragma once

#include <unordered_map>
#include <stdint.h>

#include "madevent/madcode.h"
#include "madevent/backend/tensor.h"
#include "LHAPDF/LHAPDF.h"

namespace madevent {

struct SubProcessInfo {
    uint64_t matrix_element_count;
    uint8_t on_gpu;
    uint64_t particle_count;
    uint64_t* diagram_counts;
};

class MatrixElement {
public:
    MatrixElement(
        const std::string& file,
        const std::string& param_card,
        std::size_t process_index,
        double alpha_s
    );
    MatrixElement(MatrixElement&&) noexcept = default;
    MatrixElement& operator=(MatrixElement&&) noexcept = default;
    MatrixElement(const MatrixElement&) = delete;
    MatrixElement& operator=(const MatrixElement&) = delete;
    //~MatrixElement();
    void call(Tensor momenta_in, Tensor matrix_element_out) const;
    void call_multichannel(
        Tensor momenta_in,
        Tensor amp2_remap_in,
        Tensor matrix_element_out,
        Tensor channel_weights_out
    ) const;
    bool on_gpu() const { return _on_gpu; }
    std::size_t particle_count() const { return _particle_count; }
    std::size_t diagram_count() const { return _diagram_count; }

private:
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    bool _on_gpu;
    std::size_t _particle_count;
    std::size_t _diagram_count;
    void* (*_init_subprocess)(uint64_t, const char*, double);
    void (*_compute_matrix_element)(
        void*, uint64_t, uint64_t, const double*, double*
    );
    void (*_compute_matrix_element_multichannel)(
        void*, uint64_t, uint64_t, uint64_t, const double*, const int64_t*, double*, double*
    );
    void (*_free_subprocess)(void*);
    std::vector<std::unique_ptr<void, std::function<void(void*)>>> _process_instances;
};

class PdfSet {
public:
    PdfSet(const std::string& name, int index);
    PdfSet(PdfSet&& other) noexcept = default;/* : pdf(other.pdf) {
        other.pdf = nullptr;
    }*/
    PdfSet& operator=(PdfSet&& other) noexcept = default; /* {
        pdf = other.pdf;
        other.pdf = nullptr;
        return *this;
    }*/
    PdfSet(const PdfSet&) = delete;
    PdfSet& operator=(const PdfSet&) = delete;
    //~PdfSet() { delete pdf; }
    void call(Tensor x_in, Tensor q2_in, Tensor pid_in, Tensor pdf_out) const;
    double alpha_s(double q2) const { return pdf->alphasQ2(q2); }

private:
    std::unique_ptr<LHAPDF::PDF> pdf;
};

class Context {
    /**
     * Contains global variables, loaded PDF set and matrix elements
     */
public:
    Context() : _device(cpu_device()) {}
    Context(DevicePtr device) : _device(device) {}
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    std::size_t load_matrix_element(
        const std::string& file,
        const std::string& param_card,
        std::size_t process_index,
        double alpha_s
    );
    void load_pdf(const std::string& name, int index=0);
    void define_global(
        const std::string& name,
        DataType dtype,
        const SizeVec& shape,
        bool requires_grad=false
    );
    Tensor global(const std::string& name);
    bool global_requires_grad(const std::string& name);
    const MatrixElement& matrix_element(std::size_t index) const;
    const PdfSet& pdf_set() const;
    void save(const std::string& file) const;
    void load(const std::string& file);
    DevicePtr device() { return _device; }
    static std::shared_ptr<Context> default_context();

private:
    DevicePtr _device;
    std::unordered_map<std::string, std::tuple<Tensor, bool>> globals;
    std::vector<MatrixElement> matrix_elements;
    std::optional<PdfSet> _pdf_set;
};

using ContextPtr = std::shared_ptr<Context>;

}
