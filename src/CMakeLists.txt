add_library(engine_core
  bench/chat_prompts.cpp
  bench/csv.cpp
  bench/metrics.cpp
  bench/trace.cpp
  bench/workload.cpp
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
  sched/paged_batch.cpp
  sched/request.cpp
  sched/scheduler.cpp
  tokenizer/chat.cpp
  tokenizer/text_stream.cpp
  tokenizer/tokenizer.cpp
)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json PRIVATE engine_sentencepiece)

add_library(engine_server
  server/chat_api.cpp
  server/cors.cpp
)
target_link_libraries(engine_server PUBLIC engine_core)

add_executable(engine app/main.cpp app/bench_command.cpp app/checksum_command.cpp
  app/generate_command.cpp)
target_compile_definitions(engine PRIVATE ENGINE_HAS_CUDA=$<BOOL:${ENGINE_CUDA}>
  ENGINE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_link_libraries(engine PRIVATE engine_core CLI11::CLI11)

if(ENGINE_CUDA)
  add_library(engine_kernels
    kernels/add.cu
    kernels/add_rmsnorm.cu
    kernels/attention.cu
    kernels/embedding.cu
    kernels/gemm.cpp
    kernels/paged_attention.cu
    kernels/paged_attention_v2.cu
    kernels/paged_attention_split.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/rope_fast.cu
    kernels/swiglu.cu
    kernels/swiglu_fast.cu
    kernels/write_kv.cu
    kernels/smoke.cu
    sampling/greedy.cu
    sampling/sample.cu
  )
  target_include_directories(engine_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(engine_kernels PUBLIC engine_core CUDA::cudart CUDA::cublas)

  add_library(engine_runtime
    runtime/bench.cpp
    runtime/checksum.cpp
    runtime/decode_graphs.cpp
    runtime/device_buffer.cpp
    runtime/generate.cpp
    runtime/kv_cache.cpp
    runtime/model_runner.cpp
    runtime/paged_runner.cpp
    runtime/pinned_buffer.cpp
    runtime/sampler.cpp
    runtime/serve.cpp
    runtime/weights.cpp
  )
  target_include_directories(engine_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(engine_runtime PUBLIC engine_kernels)
  target_link_libraries(engine PRIVATE engine_runtime)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/practice/CMakeLists.txt")
  add_subdirectory(practice)
endif()