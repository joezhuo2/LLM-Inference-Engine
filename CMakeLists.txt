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

include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  FIND_PACKAGE_ARGS NAMES nlohmann_json)
FetchContent_Declare(cli11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
  FIND_PACKAGE_ARGS NAMES CLI11)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest nlohmann_json cli11)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(sentencepiece QUIET IMPORTED_TARGET GLOBAL sentencepiece)
endif()
if(sentencepiece_FOUND)
  add_library(engine_sentencepiece ALIAS PkgConfig::sentencepiece)
else()
  set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
  set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(sentencepiece
    URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.1.tar.gz
    URL_HASH SHA256=c1a59e9259c9653ad0ade653dadff074cd31f0a6ff2a11316f67bee4189a8f1b
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(sentencepiece)
  target_include_directories(sentencepiece-static INTERFACE ${sentencepiece_SOURCE_DIR}/src)
  target_compile_options(sentencepiece-static PRIVATE -w)
  add_library(engine_sentencepiece ALIAS sentencepiece-static)
endif()

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-relaxed-constexpr;-Xcompiler=-Wall>"
)

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
