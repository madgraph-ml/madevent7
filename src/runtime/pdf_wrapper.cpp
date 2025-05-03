#include "madevent/runtime/pdf_wrapper.h"

#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 1
#endif
#include "LHAPDF/LHAPDF.h"

using namespace madevent;

PdfWrapper::PdfWrapper(const char* name, int index) {
    LHAPDF::setVerbosity(0);
    _pdf = LHAPDF::mkPDF(name, index);
    if (_pdf == nullptr) {
        throw std::invalid_argument("Could not load PDF");
        /*throw std::invalid_argument(std::format(
            "Could not load PDF {}, member {}", name, index
        ));*/
    }
}

double PdfWrapper::xfxQ2(int pid, double x, double q2) const {
    return static_cast<LHAPDF::PDF*>(_pdf)->xfxQ2(pid, x, q2);
}

double PdfWrapper::alphasQ2(double q2) const {
    return static_cast<LHAPDF::PDF*>(_pdf)->alphasQ2(q2);
}

PdfWrapper::~PdfWrapper() {
    delete static_cast<LHAPDF::PDF*>(_pdf);
}
