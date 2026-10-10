#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>

#include "engine/sampling/sample.h"
#include "kernels/cuda_check.h"
#include "kernels/require.h"
#include "sampling/argmax.cuh"

namespace engine {

namespace {

__device__ float gumbel(uint64_t seed, uint64_t offset) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, offset, &state);
    const double u = (double(curand(&state)) + 0.5) * 0x1p-32;
    return float(-log(-log(u)));
}

__device__ uint32_t order_key(float x) {
    const uint32_t bits = __float_as_uint(x + 0.0f);
    return bits >> 31 ? ~bits : bits | 0x80000000u;
}

template <typename T>
__device__ T block_sum(T v) {
    static_assert(kArgmaxThreads == 32 * 32);
    __shared__ T partial[32];
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffffu, v, offset);
    __syncthreads();
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
    __syncthreads();
    v = partial[threadIdx.x % 32];
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

template <typename Score>
__device__ uint32_t top_k_threshold(int vocab, int k, Score score) {
    uint32_t lo = 0;
    uint32_t hi = UINT32_MAX;
    while (lo < hi) {
        const uint32_t mid = hi - (hi - lo) / 2;
        int count = 0;
        for (int i = threadIdx.x; i < vocab; i += kArgmaxThreads)
            count += order_key(score(i)) >= mid;
        if (block_sum(count) >= k)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

__global__ void sample_kernel(int32_t* ids, const __nv_bfloat16* logits, int vocab,
                              const SampleRow* rows) {
    const __nv_bfloat16* row = logits + int64_t(blockIdx.x) * vocab;
    const SampleRow r = rows[blockIdx.x];
    const float temperature = r.params.temperature;
    const uint64_t first = r.step * uint64_t(vocab);
    const auto scaled = [&](int i) { return __bfloat162float(row[i]) / temperature; };
    uint32_t threshold = 0;
    if (temperature > 0 && r.params.top_k > 0 && r.params.top_k < vocab)
        threshold = top_k_threshold(vocab, r.params.top_k, scaled);
    const int arg = block_argmax(vocab, [&](int i) {
        if (temperature == 0) return __bfloat162float(row[i]);
        const float s = scaled(i);
        return order_key(s) >= threshold ? s + gumbel(r.params.seed, first + i) : -INFINITY;
    });
    if (threadIdx.x == 0) ids[blockIdx.x] = arg;
}

}  // namespace

void sample(const Tensor& ids, const Tensor& logits, const SampleRow* rows, Stream stream) {
    require(logits.ndim == 2 && logits.dtype == DType::BF16 && logits.shape[1] > 0, "sample",
            "logits must be BF16 [T, vocab] with vocab > 0");
    require(ids.ndim == 1 && ids.dtype == DType::I32 && ids.shape[0] == logits.shape[0], "sample",
            "ids must be I32 [T]");
    const int64_t tokens = logits.shape[0];
    if (tokens == 0) return;
    sample_kernel<<<unsigned(tokens), kArgmaxThreads, 0, stream>>>(
        static_cast<int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(logits.data),
        int(logits.shape[1]), rows);
    KERNEL_CHECK(stream);
}

}  // namespace engine
