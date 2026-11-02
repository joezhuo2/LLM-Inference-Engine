include(GoogleTest)

set(ENGINE_TEST_DEFINITIONS
  ENGINE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tinyllama"
  ENGINE_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  ENGINE_SCRIPTS_DIR="${PROJECT_SOURCE_DIR}/scripts")

add_executable(unit_tests
  unit/test_bf16.cpp
  unit/test_bench_metrics.cpp
  unit/test_broadcaster.cpp
  unit/test_chat_prompts.cpp
  unit/test_block_manager.cpp
  unit/test_chat.cpp
  unit/test_chat_api.cpp
  unit/test_checksum.cpp
  unit/test_checksum_golden.cpp
  unit/test_config.cpp
  unit/test_cors.cpp
  unit/test_kv_cache_size.cpp
  unit/test_mapped_file.cpp
  unit/test_metrics_window.cpp
  unit/test_model_weights.cpp
  unit/test_paged_batch.cpp
  unit/test_reference_golden.cpp
  unit/test_request.cpp
  unit/test_safetensors.cpp
  unit/test_scheduler.cpp
  unit/test_sampling_params.cpp
  unit/test_stop_matcher.cpp
  unit/test_tensor.cpp
  unit/test_text_stream.cpp
  unit/test_tokenizer.cpp
  unit/test_tokenizer_golden.cpp
  unit/test_trace.cpp
  unit/test_version.cpp
  unit/test_weight_layout.cpp
  unit/test_workload.cpp
)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(unit_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
target_link_libraries(unit_tests PRIVATE engine_core engine_server GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

if(ENGINE_CUDA)
  add_executable(gpu_tests
    gpu/test_add.cpp
    gpu/test_add_rmsnorm.cpp
    gpu/test_attention.cpp
    gpu/test_bench.cpp
    gpu/test_bench_batched.cpp
    gpu/test_bf16_cuda.cpp
    gpu/test_checksum_device.cpp
    gpu/test_checksum_golden.cpp
    gpu/test_cuda_check.cpp
    gpu/test_decode_graphs.cpp
    gpu/test_device_buffer.cpp
    gpu/test_embedding.cpp
    gpu/test_forward_golden.cpp
    gpu/test_gemm.cpp
    gpu/test_generate.cpp
    gpu/test_greedy.cpp
    gpu/test_kv_cache.cpp
    gpu/test_load_weights.cpp
    gpu/test_model_runner.cpp
    gpu/test_paged_attention.cpp
    gpu/test_paged_attention_v2.cpp
    gpu/test_paged_attention_split.cpp
    gpu/test_paged_runner.cpp
    gpu/test_pinned_buffer.cpp
    gpu/test_rmsnorm.cpp
    gpu/test_rope.cpp
    gpu/test_rope_fast.cpp
    gpu/test_sample.cpp
    gpu/test_serve.cpp
    gpu/test_smoke.cpp
    gpu/test_stream.cpp
    gpu/test_swiglu.cpp
    gpu/test_swiglu_fast.cpp
    gpu/test_weight_upload.cpp
    gpu/test_write_kv.cpp
  )
  target_include_directories(gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(gpu_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
  target_link_libraries(gpu_tests PRIVATE engine_runtime GTest::gtest_main)
  gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()

add_executable(todo_unit_tests
  todo/unit/test_scheduler_cancel.cpp
)
target_link_libraries(todo_unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(todo_unit_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS todo_joe)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME PythonScripts.Unittest
    COMMAND ${Python3_EXECUTABLE} -I -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/python -p "test_*.py")
  set_tests_properties(PythonScripts.Unittest PROPERTIES LABELS python)
endif()
