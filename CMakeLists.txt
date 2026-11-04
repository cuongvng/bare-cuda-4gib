cmake_minimum_required(VERSION 3.18)  # 3.18+ required for CUDA_ARCHITECTURES

project(bare_cuda_4gib
  VERSION 0.1.0
  DESCRIPTION "Bare-metal CUDA LLM inference engine for a 4GB card"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# sm_86 = Ampere GA107 (RTX 3050 Laptop). Override on a different card:
#   cmake -B build -DLLM_CUDA_ARCH=89
set(LLM_CUDA_ARCH "86" CACHE STRING "CUDA compute capability to target (no dot)")
set(CMAKE_CUDA_ARCHITECTURES ${LLM_CUDA_ARCH})

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)  # dev-time only: builds test fixtures / offline conversion

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# -lineinfo keeps ncu/nsys able to attribute time to source lines without the
# codegen cost of full -G device debug.
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

add_subdirectory(engine)
add_subdirectory(bench)
add_subdirectory(server)

message(STATUS "bare-cuda-4gib: CUDA ${CUDAToolkit_VERSION}, arch sm_${LLM_CUDA_ARCH}, build ${CMAKE_BUILD_TYPE}")
