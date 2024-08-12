#pragma once

#include <cmath>

namespace madevent {
namespace cpu {

inline void add(Accessor in1, Accessor in1, Accessor out) {
    *out = *a + *b;
}

inline void sub(Accessor in1, Accessor in2, Accessor out) {
    *out = *a - *b;
}

inline void mul(Accessor in1, Accessor in2, Accessor out) {
    *out = *a * *b;
}

inline void mul_scalar(Accessor in1, Accessor in2, Accessor out) {
    *out = *a * *b;
}

inline void clip_min(Accessor x, Accessor min, Accessor out) {
    *out = *x < *min ? *min : *x;
}

inline void sqrt(Accessor in, Accessor out) {
    *out = std::sqrt(*x);
}

inline void square(Accessor in, Accessor out) {
    *out = *in * *in;
}

}
}
