#pragma once

#include <unordered_map>

#include "madevent/madcode.h"
#include "madevent/backend/tensor.h"
#include "LHAPDF/LHAPDF.h"

namespace madevent {

class MatrixElement {
public:
    MatrixElement(std::string file, std::string param_card, std::size_t process_index);
    ~MatrixElement();
    void call(
        Tensor momenta_in,
        Tensor amp2_remap_in,
        Tensor matrix_element_out,
        Tensor channel_weights_out
    ) const;

private:
    int process_index;
    void* shared_lib;
    void* (*process_init)(const char*);
    void (*compute_me2)(void*, double*, long long*, double*, double*, int, int);
    void (*process_free)(void*);
    std::vector<void*> process_instances;
};

class PdfSet {
public:
    PdfSet(std::string name, int index);
    ~PdfSet();
    void call(Tensor x_in, Tensor q2_in, Tensor pid_in, Tensor pdf_out) const;

private:
    LHAPDF::PDF* pdf;
};

class Context {
    /**
     * Contains global variables, loaded PDF set and matrix elements
     */
public:
    Context(Device& device) : device(device) {}
    void load_matrix_element(
        std::string file, std::string param_card, std::size_t process_index
    );
    void load_pdf(std::string name, int index=0);
    void define_global(std::string name, DataType dtype, const SizeVec& shape);
    Tensor global(std::string name);
    const MatrixElement& matrix_element(std::size_t index) const;
    const PdfSet& pdf_set() const;
    void save(std::string file) const;
    void load(std::string file);
    static Context& default_context();
private:
    Device& device;
    std::unordered_map<std::string, Tensor> globals;
    std::vector<MatrixElement> matrix_elements;
    std::optional<PdfSet> _pdf_set;
};

}
