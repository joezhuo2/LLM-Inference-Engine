add_library(engine_core
  core/tensor.cpp
  core/version.cpp
  kv/cache_size.cpp
  kv/block_manager.cpp
  loader/checksum.cpp
  loader/mapped_file.cpp
  loader/safetensors.cpp
  loader/weight_layout.cpp
  model/config.cpp
  model/weights.cpp
  sampling/params.cpp
  sched/request.cpp
  tokenizer/chat.cpp
  tokenizer/text_stream.cpp
  tokenizer/tokenizer.cpp
)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json PRIVATE engine_sentencepiece)

add_executable(engine app/main.cpp app/checksum_command.cpp app/generate_command.cpp)
target_compile_definitions(engine PRIVATE ENGINE_HAS_CUDA=$<BOOL:${ENGINE_CUDA}>)
target_link_libraries(engine PRIVATE engine_core CLI11::CLI11)

if(ENGINE_CUDA)
  add_library(engine_kernels
    kernels/add.cu
    kernels/attention.cu
    kernels/embedding.cu
    kernels/gemm.cpp
    kernels/paged_attention.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/swiglu.cu
    kernels/write_kv.cu
    kernels/smoke.cu
    sampling/greedy.cu
    sampling/sample.cu
  )
  target_include_directories(engine_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(engine_kernels PUBLIC engine_core CUDA::cudart CUDA::cublas)

  add_library(engine_runtime
    runtime/checksum.cpp
    runtime/device_buffer.cpp
    runtime/generate.cpp
    runtime/kv_cache.cpp
    runtime/model_runner.cpp
    runtime/pinned_buffer.cpp
    runtime/sampler.cpp
    runtime/weights.cpp
  )
  target_include_directories(engine_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(engine_runtime PUBLIC engine_kernels)
  target_link_libraries(engine PRIVATE engine_runtime)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/practice/CMakeLists.txt")
  add_subdirectory(practice)
endif()