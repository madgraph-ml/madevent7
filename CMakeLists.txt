cmake_minimum_required (VERSION 3.15...3.27)
project (
    madevent
    VERSION 0.1
    LANGUAGES CXX #CUDA
)

set(CMAKE_POSITION_INDEPENDENT_CODE 1)

# Doesn't work for no apparent reason
#execute_process(
#    COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
#    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
#)
#list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
    #COMMAND "python" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
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

if(USE_SIMD)
    find_package(sleef CONFIG REQUIRED)
endif()
#find_package(OpenBLAS REQUIRED)

add_library(
    madevent
    #SHARED
    STATIC
    src/backend/common/invariants.h
    src/backend/common/kernels.h
    src/backend/common/kinematics.h
    src/backend/common/math.h
    src/backend/common/two_particle.h
    src/backend/common/rambo.h
    src/backend/common/cuts.h
    src/backend/common/chili.h
    src/backend/tensor.cpp
    src/backend/cpu/runtime.cpp
    src/backend/cpu/thread_pool.cpp
    src/backend/cpu/kernels.h
    src/backend/cpu/runtime_mixin.h
    src/backend/cpu/tensor.h
    src/backend/cpu/simd.h
    src/backend/cpu/simd_arm.h
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
    include/madevent/backend/tensor.h
    include/madevent/backend/cpu/runtime.h
    include/madevent/backend/cpu/thread_pool.h
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

    #src/backend/cuda/runtime.cu
    #src/backend/cuda/device.cu
    #src/backend/cuda/kernels.h
    #include/madevent/backend/cuda/runtime.h
    #include/madevent/backend/cuda/tensor.h
    #include/madevent/backend/cuda/device.h
)
target_include_directories(madevent PUBLIC include)
target_compile_features(madevent PUBLIC cxx_std_23)
if(USE_SIMD)
    target_link_libraries(madevent PUBLIC sleef::sleef)
    target_compile_definitions(madevent PRIVATE USE_SIMD=1)
endif()
#target_link_libraries(madevent PUBLIC ${OpenBLAS_LIBRARIES})
#target_include_directories(madevent PUBLIC ${OpenBLAS_INCLUDE_DIRS})
#target_compile_definitions(madevent PRIVATE CUDA_FOUND=1)

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

#target_compile_definitions(madevent_py PRIVATE CUDA_FOUND=1)
if(TORCH_FOUND)
    target_compile_definitions(_madevent_py PRIVATE TORCH_FOUND=1)
    target_link_libraries(_madevent_py PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()

target_link_libraries(_madevent_py PUBLIC madevent)
set_target_properties(_madevent_py PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS _madevent_py DESTINATION madevent7)
#install(TARGETS madevent DESTINATION madevent7/lib)

#install(TARGETS _madevent_py DESTINATION madevent7)
#install(TARGETS madevent DESTINATION ${SKBUILD_PLATLIB_DIR})
