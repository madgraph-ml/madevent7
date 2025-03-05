#include "madevent/backend/context.h"

#include <dlfcn.h>

#include "madevent/backend/cpu/thread_pool.h"

using namespace madevent;

MatrixElement::MatrixElement(
    std::string file, std::string param_card, std::size_t process_index
) :
    process_index(process_index)
{
    shared_lib = dlopen(file.c_str(), RTLD_NOW);
    if (shared_lib == nullptr) {
        throw std::runtime_error(std::format(
            "Could not load shared object {}", file
        ));
    }
    process_init = reinterpret_cast<decltype(process_init)>(
        dlsym(shared_lib, "process_init")
    );
    if (process_init == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol process_init in shared object {}", file
        ));
    }
    compute_me2 = reinterpret_cast<decltype(compute_me2)>(
        dlsym(shared_lib, "compute_me2_single")
    );
    if (compute_me2 == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol compute_me2_single in shared object {}", file
        ));
    }
    process_free = reinterpret_cast<decltype(process_free)>(
        dlsym(shared_lib, "process_free")
    );
    if (process_free == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol process_free in shared object {}", file
        ));
    }
    std::size_t thread_count = cpu::ThreadPool::instance().get_thread_count();
    for (int i = 0; i == 0 || i < thread_count; ++i) {
        process_instances.push_back(process_init(param_card.c_str()));
    }
}

MatrixElement::~MatrixElement() {
    for (auto inst : process_instances) {
        process_free(inst);
    }
    dlclose(shared_lib);
}

void MatrixElement::call(
    Tensor momenta_in,
    Tensor amp2_remap_in,
    Tensor matrix_element_out,
    Tensor channel_weights_out
) const {
    // TODO: very hacky - only works for contiguous tensor
    std::size_t batch_size = momenta_in.size(0);
    cpu::ThreadPool::instance().parallel_for<cpu::ThreadPool::pass_thread_id>(
        [&](std::size_t index, int thread_id) {
            compute_me2(
                process_instances[thread_id],
                static_cast<double*>(momenta_in.data()),
                static_cast<long long*>(amp2_remap_in.data()),
                static_cast<double*>(matrix_element_out.data()),
                static_cast<double*>(channel_weights_out.data()),
                batch_size,
                index
            );
        },
        batch_size
    );
}

PdfSet::PdfSet(std::string name, int index) {
    pdf = LHAPDF::mkPDF(name, index);
    if (pdf == nullptr) {
        throw std::invalid_argument(std::format(
            "Could not load PDF {}, member {}", name, index
        ));
    }
}

PdfSet::~PdfSet() {
    delete pdf;
}

void PdfSet::call(Tensor x_in, Tensor q2_in, Tensor pid_in, Tensor pdf_out) const {
    auto x_view = x_in.view<double, 1>();
    auto q2_view = q2_in.view<double, 1>();
    auto pid_view = pid_in.view<long long, 1>();
    auto pdf_view = pdf_out.view<double, 1>();
    madevent::cpu::ThreadPool::instance().parallel_for([&](std::size_t i) {
        pdf_view[i] = pdf->xfxQ2(pid_view[i], x_view[i], q2_view[i]);
    }, x_view.size());
}

void Context::load_matrix_element(
    std::string file, std::string param_card, std::size_t process_index
) {
    matrix_elements.emplace_back(file, param_card, process_index);
}

void Context::load_pdf(std::string name, int index) {
    _pdf_set.emplace(name, index);
}

void Context::define_global(
    std::string name, DataType dtype, const SizeVec& shape, bool requires_grad
) {
    SizeVec full_shape {1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (globals.contains(name)) {
        throw std::invalid_argument(std::format(
            "Context already contains a global named {}", name
        ));
    }
    globals[name] = {Tensor(dtype, shape, _device), requires_grad};
}

Tensor Context::global(std::string name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return std::get<0>(search->second);
    } else {
        throw std::invalid_argument(std::format(
            "Context does not contain a global named {}", name
        ));
    }
}

bool Context::global_requires_grad(std::string name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return std::get<1>(search->second);
    } else {
        throw std::invalid_argument(std::format(
            "Context does not contain a global named {}", name
        ));
    }
}

const MatrixElement& Context::matrix_element(std::size_t index) const {
    if (index >= matrix_elements.size()) {
        throw std::runtime_error("Matrix element index out of bounds");
    }
    return matrix_elements[index];
}

const PdfSet& Context::pdf_set() const {
    if (!_pdf_set) {
        throw std::runtime_error("No PDF set was loaded");
    }
    return *_pdf_set;
}

void Context::save(std::string file) const {

}

void Context::load(std::string file) {

}

ContextPtr Context::default_context() {
    static ContextPtr context = std::make_shared<Context>(cpu_device());
    return context;
}
