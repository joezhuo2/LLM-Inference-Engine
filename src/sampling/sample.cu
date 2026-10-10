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

__global__ void sample_kernel(int32_t* ids, const __nv_bfloat16* logits, int vocab,
                              const SampleRow* rows) {
    const __nv_bfloat16* row = logits + int64_t(blockIdx.x) * vocab;
    const SampleRow r = rows[blockIdx.x];
    const float temperature = r.params.temperature;
    const uint64_t first = r.step * uint64_t(vocab);
    const int arg = block_argmax(vocab, [&](int i) {
        const float x = __bfloat162float(row[i]);
        return temperature == 0 ? x : x / temperature + gumbel(r.params.seed, first + i);
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
