include(GoogleTest)

add_executable(unit_tests
  unit/test_version.cpp
)
target_link_libraries(unit_tests PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

if(ENGINE_CUDA)
  add_executable(gpu_tests
    gpu/test_smoke.cpp
  )
  target_link_libraries(gpu_tests PRIVATE engine_kernels GTest::gtest_main)
  gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()
