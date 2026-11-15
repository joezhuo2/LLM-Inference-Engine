#include <cuda_bf16.h>

#include "cuda_check.h"
#include "engine/kernels/all_reduce.h"
#include "require.h"

namespace engine {

namespace {

constexpr int kThreads = 256;

struct Partials {
    const __nv_bfloat16* p[kMaxRanks];
};

__global__ void all_reduce_add_kernel(__nv_bfloat16* x, Partials partials, int ranks, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = __bfloat162float(partials.p[0][i]);
    for (int r = 1; r < ranks; ++r) sum += __bfloat162float(partials.p[r][i]);
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + sum);
}

}  // namespace

void all_reduce_add(const Tensor& x, std::span<const Tensor> partials, Stream stream) {
    require(!partials.empty() && partials.size() <= size_t(kMaxRanks), "all_reduce_add",
            "there must be between 1 and 8 partial results");
    require(x.ndim == 2 && x.dtype == DType::BF16, "all_reduce_add", "x must be BF16 [T, hidden]");
    Partials p{};
    for (size_t r = 0; r < partials.size(); ++r) {
        require(partials[r].dtype == DType::BF16 && partials[r].ndim == 2 &&
                    partials[r].shape == x.shape,
                "all_reduce_add", "every partial result must be BF16 with the shape of x");
        p.p[r] = static_cast<const __nv_bfloat16*>(partials[r].data);
    }
    const int64_t n = x.numel();
    if (n == 0) return;
    all_reduce_add_kernel<<<unsigned((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), p, int(partials.size()), n);
    KERNEL_CHECK(stream);
}

}  // namespace engine
