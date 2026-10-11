include(GoogleTest)

set(ENGINE_TEST_DEFINITIONS
  ENGINE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tinyllama"
  ENGINE_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

add_executable(unit_tests
  unit/test_bf16.cpp
  unit/test_chat.cpp
  unit/test_checksum.cpp
  unit/test_checksum_golden.cpp
  unit/test_config.cpp
  unit/test_kv_cache_size.cpp
  unit/test_mapped_file.cpp
  unit/test_model_weights.cpp
  unit/test_reference_golden.cpp
  unit/test_request.cpp
  unit/test_safetensors.cpp
  unit/test_sampling_params.cpp
  unit/test_tensor.cpp
  unit/test_text_stream.cpp
  unit/test_tokenizer.cpp
  unit/test_tokenizer_golden.cpp
  unit/test_version.cpp
  unit/test_weight_layout.cpp
)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(unit_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
target_link_libraries(unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

add_executable(todo_unit_tests
  todo/unit/test_block_manager.cpp
  todo/unit/test_scheduler.cpp
)
target_include_directories(todo_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(todo_unit_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
target_link_libraries(todo_unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(todo_unit_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS todo_joe)

if(ENGINE_CUDA)
  add_executable(gpu_tests
    gpu/test_add.cpp
    gpu/test_attention.cpp
    gpu/test_bf16_cuda.cpp
    gpu/test_checksum_device.cpp
    gpu/test_checksum_golden.cpp
    gpu/test_cuda_check.cpp
    gpu/test_device_buffer.cpp
    gpu/test_embedding.cpp
    gpu/test_forward_golden.cpp
    gpu/test_gemm.cpp
    gpu/test_generate.cpp
    gpu/test_greedy.cpp
    gpu/test_kv_cache.cpp
    gpu/test_load_weights.cpp
    gpu/test_model_runner.cpp
    gpu/test_pinned_buffer.cpp
    gpu/test_rmsnorm.cpp
    gpu/test_rope.cpp
    gpu/test_sample.cpp
    gpu/test_smoke.cpp
    gpu/test_stream.cpp
    gpu/test_swiglu.cpp
    gpu/test_weight_upload.cpp
  )
  target_include_directories(gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(gpu_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
  target_link_libraries(gpu_tests PRIVATE engine_runtime GTest::gtest_main)
  gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)

  add_executable(todo_gpu_tests
    todo/gpu/test_paged_attention.cpp
    todo/gpu/test_write_kv.cpp
  )
  target_include_directories(todo_gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(todo_gpu_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
  target_link_libraries(todo_gpu_tests PRIVATE engine_runtime GTest::gtest_main)
  gtest_discover_tests(todo_gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS todo_joe)
endif()
