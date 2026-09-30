include(GoogleTest)

add_executable(buffer_test
  BufferTest.cpp
)
target_link_libraries(buffer_test PRIVATE GTest::gtest_main)
gtest_discover_tests(buffer_test DISCOVERY_MODE PRE_TEST)

target_compile_options(buffer_test PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(buffer_test PRIVATE -fsanitize=address)

cmake_minimum_required(VERSION 3.20)
project(safetensors_reader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(read_header SafetensorReaderTest.cpp)
target_compile_options(read_header PRIVATE -Wall -Wextra -Wpedantic)