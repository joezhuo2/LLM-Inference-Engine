#include <gtest/gtest.h>

#include "engine/kernels/gemm.h"
#include "engine/runtime/device_buffer.h"
#include "kernels/cuda_check.h"

namespace {

constexpr size_t kWorkspaceBytes = size_t(4) << 20;

}  // namespace

TEST(Blas, BindsTheGivenStream) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    {
        engine::DeviceBuffer workspace(kWorkspaceBytes);
        engine::Blas blas(s, workspace.data(), workspace.size());
        ASSERT_NE(blas.handle(), nullptr);
        cudaStream_t bound;
        CUBLAS_CHECK(cublasGetStream(blas.handle(), &bound));
        EXPECT_EQ(bound, s);
        EXPECT_EQ(blas.stream(), s);
    }
    CUDA_CHECK(cudaStreamDestroy(s));
}
