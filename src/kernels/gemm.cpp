#include "engine/kernels/gemm.h"

#include "cuda_check.h"

namespace engine {

Blas::Blas(Stream stream, void* workspace, size_t workspace_bytes) : stream_(stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream_));
    CUBLAS_CHECK(cublasSetWorkspace(handle_, workspace, workspace_bytes));
}

Blas::~Blas() {
    CUBLAS_CHECK(cublasDestroy(handle_));
}

}  // namespace engine
