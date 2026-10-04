include(GoogleTest)

add_executable(unit_tests
  unit/test_bf16.cpp
  unit/test_tensor.cpp
  unit/test_version.cpp
)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

if(ENGINE_CUDA)
  add_executable(gpu_tests
    gpu/test_bf16_cuda.cpp
    gpu/test_cuda_check.cpp
    gpu/test_device_buffer.cpp
    gpu/test_gemm.cpp
    gpu/test_pinned_buffer.cpp
    gpu/test_smoke.cpp
    gpu/test_stream.cpp
  )
  target_include_directories(gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(gpu_tests PRIVATE engine_runtime GTest::gtest_main)
  gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()
