add_library(engine_core
  core/version.cpp
)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(engine app/main.cpp)
target_link_libraries(engine PRIVATE engine_core)

if(ENGINE_CUDA)
  add_library(engine_kernels
    kernels/smoke.cu
  )
  target_link_libraries(engine_kernels PUBLIC engine_core CUDA::cudart CUDA::cublas)
  target_link_libraries(engine PRIVATE engine_kernels)
endif()
