#pragma once

#include <cstddef>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

struct cublasContext;

namespace engine {

class Blas {
public:
    Blas(Stream stream, void* workspace, size_t workspace_bytes);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    cublasContext* handle() const { return handle_; }
    Stream stream() const { return stream_; }

private:
    cublasContext* handle_ = nullptr;
    Stream stream_ = nullptr;
};

void gemm(const Blas& blas, const Tensor& y, const Tensor& x, const Tensor& w);

}  // namespace engine
