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

__device__ float gumbel(uint64_t seed, uint64_t row, uint64_t offset) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, row, offset, &state);
    const double u = (double(curand(&state)) + 0.5) * 0x1p-32;
    return float(-log(-log(u)));
}

__global__ void sample_kernel(int32_t* ids, const __nv_bfloat16* logits, int vocab,
                              float temperature, uint64_t seed, uint64_t step) {
    const __nv_bfloat16* row = logits + int64_t(blockIdx.x) * vocab;
    const uint64_t first = step * uint64_t(vocab);
    const int arg = block_argmax(vocab, [&](int i) {
        return __bfloat162float(row[i]) / temperature + gumbel(seed, blockIdx.x, first + i);
    });
    if (threadIdx.x == 0) ids[blockIdx.x] = arg;
}

}  // namespace

void sample(const Tensor& ids, const Tensor& logits, float temperature, uint64_t seed,
            uint64_t step, Stream stream) {
    require(logits.ndim == 2 && logits.dtype == DType::BF16 && logits.shape[1] > 0, "sample",
            "logits must be BF16 [T, vocab] with vocab > 0");
    require(ids.ndim == 1 && ids.dtype == DType::I32 && ids.shape[0] == logits.shape[0], "sample",
            "ids must be I32 [T]");
    require(temperature > 0 && std::isfinite(temperature), "sample",
            "temperature must be positive and finite");
    const int64_t tokens = logits.shape[0];
    if (tokens == 0) return;
    sample_kernel<<<unsigned(tokens), kArgmaxThreads, 0, stream>>>(
        static_cast<int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(logits.data),
        int(logits.shape[1]), temperature, seed, step);
    KERNEL_CHECK(stream);
}

}  // namespace engine
