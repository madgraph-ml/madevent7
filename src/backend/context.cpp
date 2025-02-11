#include "madevent/backend/context.h"

using namespace madevent;

void Context::load_matrix_element(std::string file) {
    
}

void Context::load_pdf(std::string name, int index) {

}

void Context::define_global(std::string name, DataType dtype, const SizeVec& shape) {
    SizeVec full_shape {1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (globals.contains(name)) {
        throw std::invalid_argument(std::format(
            "Context already contains a global named {}", name
        ));
    }
    globals[name] = Tensor(dtype, shape, device);
}

Tensor& Context::get_global(std::string name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return search->second;
    } else {
        throw std::invalid_argument(std::format(
            "Context does not contain a global named {}", name
        ));
    }
}

void Context::save(std::string file) const {

}

void Context::load(std::string file) {

}
