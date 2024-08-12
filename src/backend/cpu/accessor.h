#pragma once

namespace madevent {
namespace cpu {

class Accessor {
public:
    Accessor(double* _pointer, int _stride) : pointer(_pointer), stride(_stride) {}
    //double& operator*() const { return *pointer; }
    const double& operator[](size_t index) const { return pointer[index * stride]; }
    double& operator[](size_t index) const { return pointer[index * stride]; }
    operator double() const { return *pointer; }
    operator double&() { return *pointer; }

private:
    double* pointer;
    int stride;
};

}
}
