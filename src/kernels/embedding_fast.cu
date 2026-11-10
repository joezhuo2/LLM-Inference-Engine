#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/embedding.h"
#include "require.h"

namespace engine::fast {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kVecWidth = 8;

__global__ void embedding_vec_kernel(uint4* __restrict__ out, const int32_t* __restrict__ ids,
                                     const uint4* __restrict__ table, int64_t total,
                                     int64_t vecs_per_row) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t row = idx / vecs_per_row;
    const int64_t col = idx % vecs_per_row;
    out[idx] = table[int64_t(ids[row]) * vecs_per_row + col];
}

__global__ void embedding_scalar_kernel(__nv_bfloat16* __restrict__ out,
                                        const int32_t* __restrict__ ids,
                                        const __nv_bfloat16* __restrict__ table, int64_t total,
                                        int64_t hidden) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t row = idx / hidden;
    const int64_t col = idx % hidden;
    out[idx] = table[int64_t(ids[row]) * hidden + col];
}

bool aligned16(const void* p) {
    return reinterpret_cast<uintptr_t>(p) % sizeof(uint4) == 0;
}

}  // namespace

void embedding(const Tensor& out, const Tensor& ids, const Tensor& table, Stream stream) {
    require(ids.ndim == 1 && ids.dtype == DType::I32, "embedding", "ids must be I32 [T]");
    require(table.ndim == 2 && table.dtype == DType::BF16, "embedding",
            "table must be BF16 [vocab, hidden]");
    require(out.ndim == 2 && out.dtype == DType::BF16 && out.shape[0] == ids.shape[0] &&
                out.shape[1] == table.shape[1],
            "embedding", "out must be BF16 [T, hidden]");

    const int64_t tokens = ids.shape[0];
    const int64_t hidden = table.shape[1];
    if (tokens == 0 || hidden == 0) return;

    const bool vec = hidden % kVecWidth == 0 && aligned16(out.data) && aligned16(table.data);
    const int64_t per_row = vec ? hidden / kVecWidth : hidden;
    const int64_t total = tokens * per_row;
    const auto blocks = unsigned((total + kThreads - 1) / kThreads);
    const auto* d_ids = static_cast<const int32_t*>(ids.data);
    if (vec) {
        embedding_vec_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<uint4*>(out.data), d_ids,
                                                              static_cast<const uint4*>(table.data),
                                                              total, per_row);
    } else {
        embedding_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data), d_ids,
            static_cast<const __nv_bfloat16*>(table.data), total, per_row);
    }
    KERNEL_CHECK(stream);
}

}  // namespace engine::fast
