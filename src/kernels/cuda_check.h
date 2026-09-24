#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err_ = (expr);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorName(err_), cudaGetErrorString(err_));     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
