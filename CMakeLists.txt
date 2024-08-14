cmake_minimum_required (VERSION 3.14)
project (
    madevent
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_POSITION_INDEPENDENT_CODE 1)

find_package(OpenMP)

include(FetchContent)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(backward
    GIT_REPOSITORY https://github.com/bombela/backward-cpp
    GIT_TAG master
    SYSTEM
)
FetchContent_MakeAvailable(backward)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.1
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

add_library(
    madevent SHARED
    src/backend/common/invariants.h
    src/backend/common/kernels.h
    src/backend/common/kinematics.h
    src/backend/common/math.h
    src/backend/common/two_particle.h
    src/backend/cpu/runtime.cpp
    src/backend/cpu/kernels.h
    src/madcode/function.cpp
    src/madcode/instruction.cpp
    src/madcode/instruction_set_mixin.h
    src/phasespace/mapping.cpp
    src/phasespace/invariants.cpp
    src/phasespace/two_particle.cpp
    src/phasespace/luminosity.cpp
    include/madevent/backend/cpu/runtime.h
    include/madevent/madcode.h
    include/madevent/madcode/function.h
    include/madevent/madcode/function_builder_mixin.h
    include/madevent/madcode/instruction.h
    include/madevent/phasespace.h
    include/madevent/phasespace/mapping.h
    include/madevent/phasespace/invariants.h
    include/madevent/phasespace/two_particle.h
    include/madevent/phasespace/luminosity.h
)
target_include_directories(madevent PUBLIC include)
target_compile_features(madevent PUBLIC cxx_std_17)
target_link_libraries(madevent PRIVATE fmt::fmt)
if(OpenMP_CXX_FOUND)
    target_link_libraries(madevent PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test src/test/main.cpp)
target_link_libraries(test PUBLIC madevent)
target_link_libraries(test PUBLIC Backward::Object)
target_link_libraries(test PRIVATE fmt::fmt)
