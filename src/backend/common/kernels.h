#include "madevent/constants.h"

inline constexpr double EPS = 1e-12;
inline constexpr double EPS2 = 1e-24;

template<typename T, int dim, bool is_single=false>
    using FIn = typename T::template FIn<dim, is_single>;
template<typename T, int dim, bool is_single=false>
    using IIn = typename T::template IIn<dim, is_single>;
template<typename T, int dim, bool is_single=false>
    using BIn = typename T::template BIn<dim, is_single>;
template<typename T, int dim, bool is_single=false>
    using FOut = typename T::template FOut<dim, is_single>;
template<typename T, int dim, bool is_single=false>
    using IOut = typename T::template IOut<dim, is_single>;
template<typename T, int dim, bool is_single=false>
    using BOut = typename T::template BOut<dim, is_single>;
template<typename T> using FVal = typename T::FVal;
template<typename T> using IVal = typename T::IVal;
template<typename T> using BVal = typename T::BVal;

#include "math.h"
#include "kinematics.h"
#include "two_particle.h"
#include "invariants.h"
#include "rambo.h"
#include "cuts.h"
#include "chili.h"

