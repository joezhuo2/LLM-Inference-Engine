add_library(engine_core
  core/tensor.cpp
  core/version.cpp
  loader/mapped_file.cpp
  loader/safetensors.cpp
  loader/weight_layout.cpp
  model/config.cpp
  model/weights.cpp
)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(engine app/main.cpp)
target_link_libraries(engine PRIVATE engine_core CLI11::CLI11)

if(ENGINE_CUDA)
  add_library(engine_kernels
    kernels/gemm.cpp
    kernels/smoke.cu
  )
  target_link_libraries(engine_kernels PUBLIC engine_core CUDA::cudart CUDA::cublas)

  add_library(engine_runtime
    runtime/device_buffer.cpp
    runtime/pinned_buffer.cpp
    runtime/weights.cpp
  )
  target_include_directories(engine_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(engine_runtime PUBLIC engine_kernels)
  target_link_libraries(engine PRIVATE engine_runtime)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/practice/CMakeLists.txt")
  add_subdirectory(practice)
endif()