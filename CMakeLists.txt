cmake_minimum_required (VERSION 3.15...3.27)
project (
    madevent
    VERSION 0.1
    LANGUAGES CXX
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(COMPILE_CUDA 1)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE 1)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE PT_CMAKE_PREFIX
    COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})

find_package(Torch CONFIG)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.1
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

if(USE_SIMD)
    find_package(sleef CONFIG REQUIRED)
endif()
#find_package(OpenBLAS REQUIRED)
#set(BLA_VENDOR OpenBLAS)
set(BLA_STATIC ON)
set(BLA_SIZEOF_INTEGER 4)
find_package(BLAS REQUIRED)

execute_process(
    COMMAND "lhapdf-config" "--libdir"
    OUTPUT_VARIABLE LHAPDF_LIBRARY_DIR
    COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "lhapdf-config" "--incdir"
    OUTPUT_VARIABLE LHAPDF_INCLUDE_DIRS
    COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_library(
    LHAPDF_LIBRARIES
    REQUIRED
    PATHS ${LHAPDF_LIBRARY_DIR}
    NAMES "libLHAPDF.a"
)

set(
    KERNEL_SRC
    src/kernels/autograd.h
    src/kernels/invariants.h
    src/kernels/kernels.h
    src/kernels/kinematics.h
    src/kernels/math.h
    src/kernels/rambo.h
    src/kernels/cuts.h
    src/kernels/chili.h
    src/kernels/nn.h
    src/kernels/discrete.h
    src/kernels/vegas.h
    src/kernels/definitions.h
)


if(COMPILE_CUDA)
    add_library(
        madevent_cuda
        SHARED
        src/cuda/runtime.cu
        src/cuda/device.cu
        src/cuda/kernels.h
        include/madevent/cuda/runtime.h
        include/madevent/cuda/tensor.h
        include/madevent/cuda/device.h
        ${KERNEL_SRC}
    )
endif()

add_library(
    madevent_cpu
    STATIC
    src/cpu/runtime.cpp
    src/cpu/kernel_definitions.h
    src/cpu/runtime_mixin.h
    src/cpu/tensor.h
    src/cpu/simd.h
    src/cpu/simd_arm.h
    ${KERNEL_SRC}
    include/madevent/cpu/runtime.h
)
target_include_directories(madevent_cpu PUBLIC include)
target_compile_features(madevent_cpu PUBLIC cxx_std_23)
target_link_libraries(madevent_cpu PRIVATE BLAS::BLAS)
target_link_libraries(madevent_cpu PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(madevent_cpu PUBLIC ${LHAPDF_INCLUDE_DIRS})
target_link_libraries(madevent_cpu PRIVATE ${LHAPDF_LIBRARIES})
#target_compile_options(madevent_cpu PRIVATE -ffast-math)
if(USE_SIMD)
    target_link_libraries(madevent_cpu PRIVATE sleef::sleef)
    target_compile_definitions(madevent_cpu PRIVATE USE_SIMD=1)
endif()

add_library(
    madevent
    SHARED
    src/util.cpp
    src/runtime/tensor.cpp
    src/runtime/context.cpp
    src/runtime/generate.cpp
    src/runtime/thread_pool.cpp
    src/runtime/io.cpp
    src/runtime/vegas.cpp
    src/madcode/type.cpp
    src/madcode/function.cpp
    src/madcode/instruction.cpp
    src/madcode/optimizer.cpp
    src/madcode/instruction_set_mixin.h
    src/phasespace/mapping.cpp
    src/phasespace/invariants.cpp
    src/phasespace/two_particle.cpp
    src/phasespace/luminosity.cpp
    src/phasespace/topology.cpp
    src/phasespace/t_propagator_mapping.cpp
    src/phasespace/phasespace.cpp
    src/phasespace/multichannel.cpp
    src/phasespace/rambo.cpp
    src/phasespace/cuts.cpp
    src/phasespace/chili.cpp
    src/phasespace/integrand.cpp
    src/phasespace/vegas.cpp
    src/phasespace/mlp.cpp
    src/phasespace/flow.cpp
    src/phasespace/discrete_flow.cpp
    src/phasespace/discrete_sampler.cpp
    include/madevent/util.h
    include/madevent/runtime.h
    include/madevent/runtime/tensor.h
    include/madevent/runtime/context.h
    include/madevent/runtime/thread_pool.h
    include/madevent/runtime/generate.h
    include/madevent/runtime/io.h
    include/madevent/runtime/vegas.h
    include/madevent/madcode.h
    include/madevent/madcode/type.h
    include/madevent/madcode/function.h
    include/madevent/madcode/function_builder_mixin.h
    include/madevent/madcode/opcode_mixin.h
    include/madevent/madcode/instruction.h
    include/madevent/madcode/optimizer.h
    include/madevent/phasespace.h
    include/madevent/phasespace/mapping.h
    include/madevent/phasespace/invariants.h
    include/madevent/phasespace/two_particle.h
    include/madevent/phasespace/luminosity.h
    include/madevent/phasespace/topology.h
    include/madevent/phasespace/t_propagator_mapping.h
    include/madevent/phasespace/phasespace.h
    include/madevent/phasespace/multichannel.h
    include/madevent/phasespace/rambo.h
    include/madevent/phasespace/cuts.h
    include/madevent/phasespace/chili.h
    include/madevent/phasespace/integrand.h
    include/madevent/phasespace/vegas.h
    include/madevent/phasespace/mlp.h
    include/madevent/phasespace/flow.h
    include/madevent/phasespace/discrete_flow.h
    include/madevent/phasespace/discrete_sampler.h
)

target_include_directories(madevent PUBLIC include)
target_compile_features(madevent PUBLIC cxx_std_23)
target_include_directories(madevent PUBLIC ${LHAPDF_INCLUDE_DIRS})
target_link_libraries(madevent PRIVATE ${LHAPDF_LIBRARIES})
#target_link_libraries(madevent PRIVATE ${OpenBLAS_LIBRARIES})
target_link_libraries(madevent PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(madevent PRIVATE madevent_cpu)
#target_include_directories(madevent PUBLIC ${OpenBLAS_INCLUDE_DIRS})
#target_compile_definitions(madevent PRIVATE CUDA_FOUND=1)
#target_link_libraries(madevent PRIVATE -static-libstdc++)

#pybind11_add_module(
#    _madevent_py
#    src/python/madevent.cpp
#    src/python/function.cpp
#    src/python/instruction_set.h
#    src/python/function.h
#)
python_add_library(
    _madevent_py
    MODULE
    src/python/madevent.cpp
    src/python/function.cpp
    src/python/instruction_set.h
    src/python/function.h
    WITH_SOABI
)
target_link_libraries(_madevent_py PRIVATE pybind11::headers)
target_link_libraries(_madevent_py PRIVATE nlohmann_json::nlohmann_json)
#target_link_libraries(_madevent_py PRIVATE -static-libstdc++)

if(COMPILE_CUDA)
    target_compile_definitions(madevent_py PRIVATE CUDA_FOUND=1)
endif()

if(TORCH_FOUND)
    target_compile_definitions(_madevent_py PRIVATE TORCH_FOUND=1)
    target_link_libraries(_madevent_py PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()

target_link_libraries(_madevent_py PUBLIC madevent)
set_target_properties(_madevent_py PROPERTIES INSTALL_RPATH "$ORIGIN")

install(
    TARGETS madevent madevent_cpu
    DESTINATION madevent7/lib
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "madevent7/include"
    FILES_MATCHING PATTERN "*.h*"
)
install(
    TARGETS _madevent_py
    DESTINATION madevent7
)
#install(TARGETS madevent DESTINATION madevent7/lib)

#install(TARGETS _madevent_py DESTINATION madevent7)
#install(TARGETS madevent DESTINATION ${SKBUILD_PLATLIB_DIR})
