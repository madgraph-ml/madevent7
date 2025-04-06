cmake_minimum_required (VERSION 3.15...3.27)
project (
    madevent
    VERSION 0.1
    LANGUAGES CXX #CUDA
)

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

add_library(
    madevent
    SHARED
    src/util.cpp
    src/backend/common/invariants.h
    src/backend/common/kernels.h
    src/backend/common/kinematics.h
    src/backend/common/math.h
    src/backend/common/rambo.h
    src/backend/common/cuts.h
    src/backend/common/chili.h
    src/backend/common/nn.h
    src/backend/common/discrete.h
    src/backend/common/vegas.h
    src/backend/tensor.cpp
    src/backend/context.cpp
    src/backend/cpu/runtime.cpp
    src/backend/cpu/thread_pool.cpp
    src/backend/cpu/kernels.h
    src/backend/cpu/runtime_mixin.h
    src/backend/cpu/tensor.h
    src/backend/cpu/simd.h
    src/backend/cpu/simd_arm.h
    src/driver/combine.cpp
    src/driver/generate.cpp
    src/driver/io.cpp
    src/driver/vegas.cpp
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
    include/madevent/util.h
    include/madevent/backend/tensor.h
    include/madevent/backend/context.h
    include/madevent/backend/cpu/runtime.h
    include/madevent/backend/cpu/thread_pool.h
    include/madevent/driver.h
    include/madevent/driver/combine.h
    include/madevent/driver/generate.h
    include/madevent/driver/io.h
    include/madevent/driver/vegas.h
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

    #src/backend/cuda/runtime.cu
    #src/backend/cuda/device.cu
    #src/backend/cuda/kernels.h
    #include/madevent/backend/cuda/runtime.h
    #include/madevent/backend/cuda/tensor.h
    #include/madevent/backend/cuda/device.h
)

target_include_directories(madevent PUBLIC include)
target_compile_features(madevent PUBLIC cxx_std_23)
#target_compile_options(madevent PRIVATE -ffast-math)
if(USE_SIMD)
    target_link_libraries(madevent PRIVATE sleef::sleef)
    target_compile_definitions(madevent PRIVATE USE_SIMD=1)
endif()
target_include_directories(madevent PUBLIC ${LHAPDF_INCLUDE_DIRS})
target_link_libraries(madevent PRIVATE ${LHAPDF_LIBRARIES})
#target_link_libraries(madevent PRIVATE ${OpenBLAS_LIBRARIES})
target_link_libraries(madevent PRIVATE BLAS::BLAS)
target_link_libraries(madevent PRIVATE nlohmann_json::nlohmann_json)
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

#target_compile_definitions(madevent_py PRIVATE CUDA_FOUND=1)
if(TORCH_FOUND)
    target_compile_definitions(_madevent_py PRIVATE TORCH_FOUND=1)
    target_link_libraries(_madevent_py PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()

target_link_libraries(_madevent_py PUBLIC madevent)
set_target_properties(_madevent_py PROPERTIES INSTALL_RPATH "$ORIGIN")

install(
    TARGETS madevent
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
