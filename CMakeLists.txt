cmake_minimum_required(VERSION 3.28)
project(llm_inference_engine VERSION 0.1.0 LANGUAGES CXX)

option(ENGINE_FORCE_CPU "Build only CPU targets even if CUDA is available" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ENGINE_CUDA OFF)
if(NOT ENGINE_FORCE_CPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(ENGINE_CUDA ON)
  endif()
endif()
message(STATUS "ENGINE_CUDA = ${ENGINE_CUDA}")

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-relaxed-constexpr;-Xcompiler=-Wall>"
)

include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  FIND_PACKAGE_ARGS NAMES nlohmann_json)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest json)

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
