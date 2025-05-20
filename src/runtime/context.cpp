#include "madevent/runtime/context.h"

#include <dlfcn.h>

#include "madevent/runtime/thread_pool.h"

using namespace madevent;

MatrixElement::MatrixElement(const std::string& file, const std::string& param_card) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(std::format(
            "Could not load shared object {}", file
        ));
    }

    SubProcessInfo* (*subprocess_info)() = reinterpret_cast<decltype(subprocess_info)>(
        dlsym(_shared_lib.get(), "subprocess_info")
    );
    if (subprocess_info == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol subprocess_info in shared object {}", file
        ));
    }
    _subprocess_info = *subprocess_info();

    _init_subprocess = reinterpret_cast<decltype(_init_subprocess)>(
        dlsym(_shared_lib.get(), "init_subprocess")
    );
    if (_init_subprocess == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol init_subprocess in shared object {}", file
        ));
    }

    _compute_matrix_element = reinterpret_cast<decltype(_compute_matrix_element)>(
        dlsym(_shared_lib.get(), "compute_matrix_element")
    );
    if (_compute_matrix_element == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol compute_matrix_element in shared object {}", file
        ));
    }

    _compute_matrix_element_multichannel = reinterpret_cast<
        decltype(_compute_matrix_element_multichannel)
    >(dlsym(_shared_lib.get(), "compute_matrix_element_multichannel"));
    if (_compute_matrix_element_multichannel == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol compute_matrix_element_multichannel in shared object {}",
            file
        ));
    }

    _free_subprocess = reinterpret_cast<decltype(_free_subprocess)>(
        dlsym(_shared_lib.get(), "free_subprocess")
    );
    if (_free_subprocess == nullptr) {
        throw std::runtime_error(std::format(
            "Did not find symbol free_subprocess in shared object {}", file
        ));
    }

    std::size_t thread_count = ThreadPool::instance().get_thread_count();
    for (int i = 0; i == 0 || i < thread_count; ++i) {
        _process_instances.push_back(
            std::unique_ptr<void, std::function<void(void*)>>(
                _init_subprocess(param_card.c_str()),
                [this](void* proc) { _free_subprocess(proc); }
            )
        );
    }
}

void MatrixElement::call(
    Tensor momenta_in, Tensor flavor_in, Tensor mirror_in, Tensor matrix_element_out
) const {
    // TODO: maybe copy can be avoided sometimes
    momenta_in = momenta_in.contiguous();
    flavor_in = flavor_in.contiguous();
    mirror_in = mirror_in.contiguous();
    // TODO: move to backend
    auto batch_size = momenta_in.size(0);
    auto input_particle_count = momenta_in.size(1);
    if (input_particle_count != particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    auto& pool = ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto flavor_ptr = static_cast<int64_t*>(flavor_in.data());
    auto mirror_ptr = static_cast<int64_t*>(mirror_in.data());
    auto me_ptr = static_cast<double*>(matrix_element_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        _compute_matrix_element(
            _process_instances[0].get(), batch_size, batch_size,
            mom_ptr, flavor_ptr, mirror_ptr, me_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            _compute_matrix_element(
                _process_instances[thread_id].get(),
                std::min(batch_size - offset, count_per_thread),
                batch_size, mom_ptr + offset, flavor_ptr + offset,
                mirror_ptr + offset, me_ptr + offset
            );
        });
    }
}

void MatrixElement::call_multichannel(
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
) const {
    // TODO: maybe copy can be avoided sometimes
    momenta_in = momenta_in.contiguous();
    alpha_s_in = alpha_s_in.contiguous();
    random_in = random_in.contiguous();
    flavor_in = flavor_in.contiguous();
    mirror_in = mirror_in.contiguous();
    amp2_remap_in = amp2_remap_in.contiguous();
    // TODO: move to backend
    auto batch_size = momenta_in.size(0);
    auto input_particle_count = momenta_in.size(1);
    auto input_diagram_count = amp2_remap_in.size(1);
    auto channel_count = channel_weights_out.size(1);
    if (input_particle_count != particle_count()) {
        throw std::runtime_error("Incompatible particle count");
    }
    if (input_diagram_count != diagram_count()) {
        throw std::runtime_error("Incompatible diagram count");
    }
    auto& pool = ThreadPool::instance();
    auto thread_count = pool.get_thread_count();
    auto mom_ptr = static_cast<double*>(momenta_in.data());
    auto alpha_ptr = static_cast<double*>(alpha_s_in.data());
    auto random_ptr = static_cast<double*>(random_in.data());
    auto flavor_ptr = static_cast<int64_t*>(flavor_in.data());
    auto mirror_ptr = static_cast<int64_t*>(mirror_in.data());
    auto remap_ptr = static_cast<int64_t*>(amp2_remap_in.data());
    auto me_ptr = static_cast<double*>(matrix_element_out.data());
    auto cw_ptr = static_cast<double*>(channel_weights_out.data());
    auto color_ptr = static_cast<int64_t*>(color_out.data());
    auto diag_ptr = static_cast<int64_t*>(diagram_out.data());
    if (thread_count == 0 || batch_size < thread_count * 100) {
        _compute_matrix_element_multichannel(
            _process_instances[0].get(), batch_size, batch_size, channel_count,
            mom_ptr, alpha_ptr, random_ptr, flavor_ptr, mirror_ptr, remap_ptr,
            me_ptr, cw_ptr, color_ptr, diag_ptr
        );
    } else {
        auto count_per_thread = (batch_size + thread_count - 1) / thread_count;
        pool.parallel([&](std::size_t thread_id) {
            std::size_t offset = thread_id * count_per_thread;
            _compute_matrix_element_multichannel(
                _process_instances[thread_id].get(),
                std::min(batch_size - offset, count_per_thread),
                batch_size, channel_count,
                mom_ptr + offset, alpha_ptr + offset, random_ptr + offset,
                flavor_ptr + offset, mirror_ptr + offset, remap_ptr + offset,
                me_ptr + offset, cw_ptr + offset, color_ptr + offset, diag_ptr + offset
            );
        });
    }
}

PdfSet::PdfSet(const std::string& name, int index) :
    pdf(name.c_str(), index) {}

void PdfSet::call(Tensor x_in, Tensor q2_in, Tensor pid_in, Tensor pdf_out) const {
    auto x_view = x_in.view<double, 1>();
    auto q2_view = q2_in.view<double, 1>();
    auto pid_view = pid_in.view<int64_t, 1>();
    auto pdf_view = pdf_out.view<double, 1>();
    ThreadPool::instance().parallel_for([&](std::size_t i) {
        pdf_view[i] = pdf.xfxQ2(pid_view[i], x_view[i], q2_view[i]);
    }, x_view.size());
}

std::size_t Context::load_matrix_element(
    const std::string& file, const std::string& param_card
) {
    matrix_elements.emplace_back(file, param_card);
    return matrix_elements.size() - 1;
}

void Context::load_pdf(const std::string& name, int index) {
    _pdf_set.emplace(name, index);
}

Tensor Context::define_global(
    const std::string& name, DataType dtype, const SizeVec& shape, bool requires_grad
) {
    SizeVec full_shape {1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (globals.contains(name)) {
        throw std::invalid_argument(std::format(
            "Context already contains a global named {}", name
        ));
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
        throw std::invalid_argument(std::format(
            "Context does not contain a global named {}", name
        ));
    }
}

bool Context::global_exists(const std::string& name) {
    return globals.find(name) != globals.end();
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

ContextPtr madevent::default_context() {
    static ContextPtr context = std::make_shared<Context>(cpu_device());
    return context;
}

ContextPtr madevent::default_cuda_context() {
    static ContextPtr context = std::make_shared<Context>(cuda_device());
    return context;
}
