#pragma once

#include <unordered_map>

#include "madevent/madcode.h"
#include "madevent/backend/tensor.h"

namespace madevent {

class MatrixElement {

};

class PdfSet {

};

class Context {
    /**
     * Contains global variables, loaded PDF set and matrix elements
     */
public:
    Context(Device& device) : device(device) {}
    void load_matrix_element(std::string file);
    void load_pdf(std::string name, int index=0);
    void define_global(std::string name, DataType dtype, const SizeVec& shape);
    Tensor& get_global(std::string name);
    void save(std::string file) const;
    void load(std::string file);
private:
    Device& device;
    std::unordered_map<std::string, Tensor> globals;
    std::vector<MatrixElement> matrix_elements;
    std::optional<PdfSet> pdf_set;
};

}
