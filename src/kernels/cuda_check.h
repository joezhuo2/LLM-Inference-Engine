#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (expr);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err_), \
                         cudaGetErrorString(err_));                                             \
            std::abort();                                                                       \
        }                                                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                                              \
    do {                                                                                \
        cublasStatus_t status_ = (expr);                                                \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,                 \
                         cublasGetStatusName(status_), cublasGetStatusString(status_)); \
            std::abort();                                                               \
        }                                                                               \
    } while (0)

// Debug builds synchronize after every launch so an asynchronous fault is reported at the kernel that caused it. Synchronizing a stream that is being captured into a CUDA Graph invalidates the capture, so the sync is skipped while capturing.
#ifdef NDEBUG
#define KERNEL_CHECK(stream) CUDA_CHECK(cudaGetLastError())
#else
#define KERNEL_CHECK(stream)                                                                    \
    do {                                                                                        \
        CUDA_CHECK(cudaGetLastError());                                                         \
        cudaStreamCaptureStatus capture_;                                                       \
        CUDA_CHECK(cudaStreamIsCapturing((stream), &capture_));                                 \
        if (capture_ == cudaStreamCaptureStatusNone) CUDA_CHECK(cudaStreamSynchronize(stream)); \
    } while (0)
#endif
