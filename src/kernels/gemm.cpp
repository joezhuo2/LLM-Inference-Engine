#include "engine/kernels/gemm.h"

#include <stdexcept>
#include <string>

#include "cuda_check.h"

namespace engine {

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("gemm: ") + what);
}

}  // namespace

Blas::Blas(Stream stream, void* workspace, size_t workspace_bytes) : stream_(stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream_));
    CUBLAS_CHECK(cublasSetWorkspace(handle_, workspace, workspace_bytes));
}

Blas::~Blas() {
    CUBLAS_CHECK(cublasDestroy(handle_));
}

void gemm(const Blas& blas, const Tensor& y, const Tensor& x, const Tensor& w) {
    require(x.ndim == 2 && w.ndim == 2 && y.ndim == 2, "x, w and y must be 2-D");
    require(x.dtype == DType::BF16 && w.dtype == DType::BF16 && y.dtype == DType::BF16,
            "x, w and y must be BF16");
    require(x.shape[1] == w.shape[1], "x must be [B, in] and w must be [out, in]");
    require(y.shape[0] == x.shape[0] && y.shape[1] == w.shape[0], "y must be [B, out]");

    const int b = int(x.shape[0]);
    const int in = int(x.shape[1]);
    const int out = int(w.shape[0]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(blas.handle(), CUBLAS_OP_T, CUBLAS_OP_N, out, b, in, &alpha, w.data,
                              CUDA_R_16BF, in, x.data, CUDA_R_16BF, in, &beta, y.data, CUDA_R_16BF,
                              out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace engine
