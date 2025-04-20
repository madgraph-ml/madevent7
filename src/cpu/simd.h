#pragma once

#ifdef USE_SIMD

#ifdef __aarch64__
#include "simd_arm.h"
#endif

#else // USE_SIMD

constexpr int simd_vec_size = 1;

#endif // USE_SIMD
