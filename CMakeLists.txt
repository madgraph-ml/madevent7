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


############################################################################################
# Load dependencies                                                                        #
############################################################################################

find_package(Python REQUIRED COMPONENTS Interpreter Development)
include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

if(COMPILE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(USE_SIMD)
    set(
        sleef_libname
        ${CMAKE_STATIC_LIBRARY_PREFIX}sleef${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    ExternalProject_Add(libsleef
        GIT_REPOSITORY https://github.com/shibatch/sleef
        GIT_TAG 3.9.0
        UPDATE_DISCONNECTED true
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DSLEEF_BUILD_TESTS=OFF
        INSTALL_BYPRODUCTS <INSTALL_DIR>/lib/${sleef_libname}
    )
    ExternalProject_Get_Property(libsleef install_dir)
    add_library(sleef::sleef INTERFACE IMPORTED GLOBAL)
    add_dependencies(sleef::sleef libsleef)
    file(MAKE_DIRECTORY ${install_dir}/include)
    target_include_directories(sleef::sleef INTERFACE ${install_dir}/include)
    target_link_libraries(
        sleef::sleef
        INTERFACE
        ${install_dir}/lib/${sleef_libname}
    )
    #find_package(sleef CONFIG REQUIRED)
endif()

#find_package(OpenBLAS REQUIRED)
#set(BLA_VENDOR OpenBLAS)
set(BLA_STATIC ON)
set(BLA_SIZEOF_INTEGER 4)
find_package(BLAS REQUIRED)

set(
    lhapdf_libname
    ${CMAKE_STATIC_LIBRARY_PREFIX}LHAPDF${CMAKE_STATIC_LIBRARY_SUFFIX}
)
ExternalProject_Add(
    lhapdf
    URL https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.5.tar.gz
    CONFIGURE_HANDLED_BY_BUILD true
    CONFIGURE_COMMAND autoreconf -f -i <SOURCE_DIR>
        && <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-pic --disable-python
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    TEST_COMMAND ""
    INSTALL_BYPRODUCTS <INSTALL_DIR>/lib/${lhapdf_libname}
)
ExternalProject_Get_Property(lhapdf install_dir)
add_library(lhapdf::lhapdf INTERFACE IMPORTED GLOBAL)
add_dependencies(lhapdf::lhapdf lhapdf)
file(MAKE_DIRECTORY ${install_dir}/include)
target_include_directories(lhapdf::lhapdf INTERFACE ${install_dir}/include)
target_link_libraries(
    lhapdf::lhapdf
    INTERFACE
    ${install_dir}/lib/${lhapdf_libname}
)


############################################################################################
# Main madevent module                                                                     #
############################################################################################

add_library(
    madevent
    SHARED
    src/runtime/format.cpp
    src/runtime/tensor.cpp
    src/runtime/context.cpp
    src/runtime/pdf_wrapper.cpp
    src/runtime/generate.cpp
    src/runtime/thread_pool.cpp
    src/runtime/io.cpp
    src/runtime/vegas.cpp
    src/runtime/runtime_base.cpp
    src/madcode/type.cpp
    src/madcode/function.cpp
    src/madcode/instruction.cpp
    src/madcode/optimizer.cpp
    src/madcode/instruction_set_mixin.h
    src/phasespace/base.cpp
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
    include/madevent/runtime/format.h
    include/madevent/runtime/tensor.h
    include/madevent/runtime/context.h
    include/madevent/runtime/pdf_wrapper.h
    include/madevent/runtime/thread_pool.h
    include/madevent/runtime/generate.h
    include/madevent/runtime/io.h
    include/madevent/runtime/vegas.h
    include/madevent/runtime/runtime_base.h
    include/madevent/madcode.h
    include/madevent/madcode/type.h
    include/madevent/madcode/function.h
    include/madevent/madcode/function_builder_mixin.h
    include/madevent/madcode/opcode_mixin.h
    include/madevent/madcode/instruction.h
    include/madevent/madcode/optimizer.h
    include/madevent/phasespace.h
    include/madevent/phasespace/base.h
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
target_compile_features(madevent PUBLIC cxx_std_20)
#target_include_directories(madevent PUBLIC ${LHAPDF_INCLUDE_DIRS})
target_link_libraries(madevent PRIVATE lhapdf::lhapdf)
target_link_libraries(madevent PUBLIC nlohmann_json::nlohmann_json)
#target_link_libraries(madevent PRIVATE -static-libstdc++)

install(TARGETS madevent DESTINATION madevent7/lib)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "madevent7/include"
    FILES_MATCHING PATTERN "*.h*"
)


############################################################################################
# Compute kernel source files                                                              #
############################################################################################

set(
    KERNEL_SRC
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
    src/kernels/operations.h
)


############################################################################################
# CUDA runtime                                                                             #
############################################################################################

if(COMPILE_CUDA)
    add_library(
        madevent_cuda
        SHARED
        src/cuda/runtime.cu
        src/cuda/device.cu
        src/cuda/kernel_definitions.h
        src/cuda/runtime.h
        src/cuda/tensor.h
        src/cuda/device.h
        ${KERNEL_SRC}
    )
    set_target_properties(
        madevent_cuda PROPERTIES CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120"
    )
    target_include_directories(madevent_cuda PUBLIC include)
    set_property(TARGET madevent_cuda PROPERTY CUDA_STANDARD 20)
    target_link_libraries(madevent_cuda PRIVATE madevent)
    target_link_libraries(madevent_cuda PRIVATE CUDA::cublas)
    target_link_libraries(madevent_cuda PRIVATE CUDA::curand)
    #target_link_libraries(madevent_cuda PRIVATE Thrust)
    install(TARGETS madevent_cuda DESTINATION madevent7/lib)
endif()


############################################################################################
# CPU runtime                                                                              #
############################################################################################

if(NOT USE_SIMD)
    set(CPU_TARGETS madevent_cpu)
    set(SIMD_FLAGS "")
    set(SIMD_DEFINES "")
elseif(APPLE)
    set(CPU_TARGETS madevent_cpu madevent_cpu_neon)
    set(SIMD_FLAGS "" "")
    set(SIMD_DEFINES "" "USE_SIMD_NEON=1")
else()
    set(CPU_TARGETS madevent_cpu madevent_cpu_avx2 madevent_cpu_avx512)
    set(SIMD_FLAGS "" "-mavx2" "-mavx512")
    set(SIMD_DEFINES "" "USE_SIMD_AVX2=1" "USE_SIMD_AVX512=1")
endif()

foreach(cpu_target simd_flag simd_define IN ZIP_LISTS CPU_TARGETS SIMD_FLAGS SIMD_DEFINES)
    add_library(
        ${cpu_target}
        SHARED
        src/cpu/runtime.h
        src/cpu/device.h
        src/cpu/runtime.cpp
        src/cpu/device.cpp
        src/cpu/kernel_definitions.h
        src/cpu/runtime_mixin.h
        src/cpu/tensor.h
        src/cpu/simd.h
        src/cpu/simd_arm.h
        src/cpu/simd_x86_256.h
        ${KERNEL_SRC}
    )
    target_include_directories(${cpu_target} PUBLIC include)
    target_compile_features(${cpu_target} PUBLIC cxx_std_20)
    target_link_libraries(${cpu_target} PRIVATE BLAS::BLAS)
    target_link_libraries(${cpu_target} PRIVATE madevent)
    if(NOT cpu_target STREQUAL "madevent_cpu")
        target_link_libraries(${cpu_target} PRIVATE sleef::sleef)
        target_compile_definitions(${cpu_target} PRIVATE ${simd_define})
        target_compile_options(${cpu_target} PRIVATE ${simd_flag})
        target_compile_definitions(${cpu_target} PRIVATE USE_SIMD=1)
    endif()
    install(TARGETS ${cpu_target} DESTINATION madevent7/lib)
endforeach()


############################################################################################
# Python bindings                                                                          #
############################################################################################

python_add_library(
    _madevent_py
    MODULE
    src/python/madevent.cpp
    src/python/instruction_set.h
    src/python/function_runtime.cpp
    src/python/function_runtime.h
    WITH_SOABI
)
target_link_libraries(_madevent_py PRIVATE pybind11::headers)
target_link_libraries(_madevent_py PUBLIC madevent)
install(TARGETS _madevent_py DESTINATION madevent7)

