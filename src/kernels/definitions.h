#pragma once

#include "madevent/constants.h"

#ifdef __CUDACC__
#include "../cuda/kernel_definitions.h"
#else
#include "../cpu/kernel_definitions.h"
#endif

namespace madevent {
namespace kernels {

inline constexpr double EPS = 1e-12;
inline constexpr double EPS2 = 1e-24;

template<typename T, int dim> using FIn = typename T::template FIn<dim>;
template<typename T, int dim> using IIn = typename T::template IIn<dim>;
template<typename T, int dim> using BIn = typename T::template BIn<dim>;
template<typename T, int dim> using FOut = typename T::template FOut<dim>;
template<typename T, int dim> using IOut = typename T::template IOut<dim>;
template<typename T, int dim> using BOut = typename T::template BOut<dim>;
template<typename T> using FVal = typename T::FVal;
template<typename T> using IVal = typename T::IVal;
template<typename T> using BVal = typename T::BVal;

}
}
