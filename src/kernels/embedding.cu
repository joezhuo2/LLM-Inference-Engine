#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/embedding.h"
#include "require.h"

namespace engine::naive {

namespace {

__global__ void embedding_kernel(__nv_bfloat16* out, const int32_t* ids, const __nv_bfloat16* table,
                                 int hidden) {
    const int64_t row = blockIdx.x;
    const __nv_bfloat16* src = table + int64_t(ids[row]) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[row * hidden + i] = src[i];
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
    if (tokens == 0) return;
    embedding_kernel<<<unsigned(tokens), 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const int32_t*>(ids.data),
        static_cast<const __nv_bfloat16*>(table.data), int(table.shape[1]));
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
