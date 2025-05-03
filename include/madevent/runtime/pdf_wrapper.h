#pragma once

namespace madevent {

class PdfWrapper {
public:
    PdfWrapper(const char* name, int index);
    PdfWrapper(PdfWrapper&&) = default;
    PdfWrapper& operator=(PdfWrapper&&) = default;
    PdfWrapper(const PdfWrapper&) = delete;
    PdfWrapper& operator=(const PdfWrapper&) = delete;
    ~PdfWrapper();
    double xfxQ2(int pid, double x, double q2) const;
    double alphasQ2(double q2) const;
private:
    void* _pdf;
};

}
