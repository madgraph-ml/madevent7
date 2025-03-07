#include "madevent/backend/context.h"

#include <dlfcn.h>

#include "madevent/backend/cpu/thread_pool.h"

using namespace madevent;

MatrixElement::MatrixElement(
    const std::string& file, const std::string& param_card, std::size_t process_index
) {
    _shared_lib = dlopen(file.c_str(), RTLD_NOW);
    if (_shared_lib == nullptr) {
        throw std::runtime_error(std::format(
            "Could not load shared object {}", file
        ));
    }

    SubProcessInfo* (*subprocess_info)() = reinterpret_cast<decltype(subprocess_info)>(
        dlsym(_shared_lib, "subprocess_info")
    );
    if (subprocess_info == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol subprocess_info in shared object {}", file
        ));
    }
    const SubProcessInfo* info = subprocess_info();
    if (process_index >= info->matrix_element_count) {
        throw std::invalid_argument("Process index out of range");
    }
    _on_gpu = info->on_gpu;
    _particle_count = info->particle_count;
    _diagram_count = info->diagram_counts[process_index];

    _init_subprocess = reinterpret_cast<decltype(_init_subprocess)>(
        dlsym(_shared_lib, "init_subprocess")
    );
    if (_init_subprocess == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol init_subprocess in shared object {}", file
        ));
    }

    _compute_matrix_element = reinterpret_cast<decltype(_compute_matrix_element)>(
        dlsym(_shared_lib, "compute_matrix_element")
    );
    if (_compute_matrix_element == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol compute_matrix_element in shared object {}", file
        ));
    }

    _compute_matrix_element_multichannel = reinterpret_cast<
        decltype(_compute_matrix_element_multichannel)
    >(dlsym(_shared_lib, "compute_matrix_element_multichannel"));
    if (_compute_matrix_element_multichannel == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol compute_matrix_element_multichannel in shared object {}", file
        ));
    }

    _free_subprocess = reinterpret_cast<decltype(_free_subprocess)>(
        dlsym(_shared_lib, "free_subprocess")
    );
    if (_free_subprocess == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol free_subprocess in shared object {}", file
        ));
    }

    std::size_t thread_count = cpu::ThreadPool::instance().get_thread_count();
    for (int i = 0; i == 0 || i < thread_count; ++i) {
        _process_instances.push_back(_init_subprocess(process_index, param_card.c_str()));
    }
}

MatrixElement::~MatrixElement() {
    for (auto inst : _process_instances) {
        _free_subprocess(inst);
    }
    dlclose(_shared_lib);
}

void MatrixElement::call(Tensor momenta_in, Tensor matrix_element_out) const {
    // TODO: maybe copy can be avoided sometimes
    momenta_in = momenta_in.contiguous();
    // TODO: move to backend
    auto batch_size = momenta_in.size(0);
    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != _particle_count) {
        throw std::runtime_error("Incompatible particle count");
    }
    auto& pool = cpu::ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto me_ptr = static_cast<double*>(matrix_element_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        _compute_matrix_element(
            _process_instances[0], batch_size, batch_size, mom_ptr, me_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            _compute_matrix_element(
                _process_instances[thread_id],
                std::min(batch_size - offset, count_per_thread),
                batch_size, mom_ptr + offset, me_ptr + offset
            );
        });
    }
}

void MatrixElement::call_multichannel(
    Tensor momenta_in,
    Tensor amp2_remap_in,
    Tensor matrix_element_out,
    Tensor channel_weights_out
) const {
    // TODO: maybe copy can be avoided sometimes
    momenta_in = momenta_in.contiguous();
    // TODO: move to backend
    auto batch_size = momenta_in.size(0);
    auto input_particle_count = momenta_in.size(1);
    auto input_diagram_count = amp2_remap_in.size(1);
    auto channel_count = channel_weights_out.size(1);
    if (input_particle_count != _particle_count) {
        throw std::runtime_error("Incompatible particle count");
    }
    if (input_diagram_count != _diagram_count) {
        throw std::runtime_error("Incompatible diagram count");
    }
    auto& pool = cpu::ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto remap_ptr = static_cast<int64_t*>(amp2_remap_in.data());
    auto me_ptr = static_cast<double*>(matrix_element_out.data());
    auto cw_ptr = static_cast<double*>(channel_weights_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        _compute_matrix_element_multichannel(
            _process_instances[0], batch_size, batch_size, channel_count,
            mom_ptr, remap_ptr, me_ptr, cw_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            _compute_matrix_element_multichannel(
                _process_instances[thread_id],
                std::min(batch_size - offset, count_per_thread),
                batch_size, channel_count,
                mom_ptr + offset, remap_ptr, me_ptr + offset, cw_ptr + offset
            );
        });
    }
}

PdfSet::PdfSet(const std::string& name, int index) {
    LHAPDF::setVerbosity(0);
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
    auto pid_view = pid_in.view<int64_t, 1>();
    auto pdf_view = pdf_out.view<double, 1>();
    madevent::cpu::ThreadPool::instance().parallel_for([&](std::size_t i) {
        pdf_view[i] = pdf->xfxQ2(pid_view[i], x_view[i], q2_view[i]);
    }, x_view.size());
}

void Context::load_matrix_element(
    const std::string& file, const std::string& param_card, std::size_t process_index
) {
    matrix_elements.emplace_back(file, param_card, process_index);
}

void Context::load_pdf(const std::string& name, int index) {
    _pdf_set.emplace(name, index);
}

void Context::define_global(
    const std::string& name, DataType dtype, const SizeVec& shape, bool requires_grad
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

Tensor Context::global(const std::string& name) {
    if (auto search = globals.find(name); search != globals.end()) {
        return std::get<0>(search->second);
    } else {
        throw std::invalid_argument(std::format(
            "Context does not contain a global named {}", name
        ));
    }
}

bool Context::global_requires_grad(const std::string& name) {
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

void Context::save(const std::string& file) const {

}

void Context::load(const std::string& file) {

}

ContextPtr Context::default_context() {
    static ContextPtr context = std::make_shared<Context>(cpu_device());
    return context;
}
