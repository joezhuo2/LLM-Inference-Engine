#include <cuda_bf16.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "engine/sampling/greedy.h"
#include "kernels/cuda_check.h"
#include "kernels/require.h"

namespace engine {

namespace {

constexpr int kThreads = 1024;

__device__ bool better(float v, int i, float best, int arg) {
    return i != INT_MAX && (arg == INT_MAX || v > best || (v == best && i < arg));
}

__global__ void greedy_kernel(int32_t* ids, const __nv_bfloat16* logits, int vocab) {
    __shared__ float values[kThreads];
    __shared__ int args[kThreads];
    const __nv_bfloat16* row = logits + int64_t(blockIdx.x) * vocab;

    float best = -INFINITY;
    int arg = INT_MAX;
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
        const float v = __bfloat162float(row[i]);
        if (better(v, i, best, arg)) {
            best = v;
            arg = i;
        }
    }
    values[threadIdx.x] = best;
    args[threadIdx.x] = arg;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const int other = threadIdx.x + stride;
            if (better(values[other], args[other], values[threadIdx.x], args[threadIdx.x])) {
                values[threadIdx.x] = values[other];
                args[threadIdx.x] = args[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[blockIdx.x] = args[0];
}

}  // namespace

void greedy(const Tensor& ids, const Tensor& logits, Stream stream) {
    require(logits.ndim == 2 && logits.dtype == DType::BF16 && logits.shape[1] > 0, "greedy",
            "logits must be BF16 [T, vocab] with vocab > 0");
    require(ids.ndim == 1 && ids.dtype == DType::I32 && ids.shape[0] == logits.shape[0], "greedy",
            "ids must be I32 [T]");
    const int64_t tokens = logits.shape[0];
    if (tokens == 0) return;
    greedy_kernel<<<unsigned(tokens), kThreads, 0, stream>>>(
        static_cast<int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(logits.data),
        int(logits.shape[1]));
    KERNEL_CHECK(stream);
}

}  // namespace engine
