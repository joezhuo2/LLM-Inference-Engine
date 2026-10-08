include(GoogleTest)

set(ENGINE_TEST_DEFINITIONS ENGINE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tinyllama")

add_executable(unit_tests
  unit/test_bf16.cpp
  unit/test_config.cpp
  unit/test_mapped_file.cpp
  unit/test_model_weights.cpp
  unit/test_tensor.cpp
  unit/test_version.cpp
  unit/test_weight_layout.cpp
)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(unit_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
target_link_libraries(unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

add_executable(todo_unit_tests
  todo/unit/test_safetensors.cpp
)
target_include_directories(todo_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(todo_unit_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
target_link_libraries(todo_unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(todo_unit_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS todo_joe)

if(ENGINE_CUDA)
  add_executable(gpu_tests
    gpu/test_bf16_cuda.cpp
    gpu/test_cuda_check.cpp
    gpu/test_device_buffer.cpp
    gpu/test_gemm.cpp
    gpu/test_pinned_buffer.cpp
    gpu/test_smoke.cpp
    gpu/test_stream.cpp
    gpu/test_weight_upload.cpp
  )
  target_include_directories(gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(gpu_tests PRIVATE ${ENGINE_TEST_DEFINITIONS})
  target_link_libraries(gpu_tests PRIVATE engine_runtime GTest::gtest_main)
  gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()
