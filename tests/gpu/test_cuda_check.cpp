#include <gtest/gtest.h>

#include "kernels/cuda_check.h"

TEST(CudaCheckDeathTest, CudaErrorAbortsWithItsName) {
    EXPECT_DEATH(CUDA_CHECK(cudaErrorInvalidValue), "cudaErrorInvalidValue");
}

TEST(CudaCheckDeathTest, CublasErrorAbortsWithItsName) {
    EXPECT_DEATH(CUBLAS_CHECK(CUBLAS_STATUS_INVALID_VALUE), "CUBLAS_STATUS_INVALID_VALUE");
}

TEST(CudaCheck, KernelCheckPassesOnACleanStream) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    KERNEL_CHECK(s);
    CUDA_CHECK(cudaStreamDestroy(s));
}
