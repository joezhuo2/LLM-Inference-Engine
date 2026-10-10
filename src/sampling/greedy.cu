#include <cuda_bf16.h>

#include <cstdint>

#include "engine/sampling/greedy.h"
#include "kernels/cuda_check.h"
#include "kernels/require.h"
#include "sampling/argmax.cuh"

namespace engine {

namespace {

__global__ void greedy_kernel(int32_t* ids, const __nv_bfloat16* logits, int vocab) {
    const __nv_bfloat16* row = logits + int64_t(blockIdx.x) * vocab;
    const int arg = block_argmax(vocab, [&](int i) { return __bfloat162float(row[i]); });
    if (threadIdx.x == 0) ids[blockIdx.x] = arg;
}

}  // namespace

void greedy(const Tensor& ids, const Tensor& logits, Stream stream) {
    require(logits.ndim == 2 && logits.dtype == DType::BF16 && logits.shape[1] > 0, "greedy",
            "logits must be BF16 [T, vocab] with vocab > 0");
    require(ids.ndim == 1 && ids.dtype == DType::I32 && ids.shape[0] == logits.shape[0], "greedy",
            "ids must be I32 [T]");
    const int64_t tokens = logits.shape[0];
    if (tokens == 0) return;
    greedy_kernel<<<unsigned(tokens), kArgmaxThreads, 0, stream>>>(
        static_cast<int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(logits.data),
        int(logits.shape[1]));
    KERNEL_CHECK(stream);
}

}  // namespace engine
