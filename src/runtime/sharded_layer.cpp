#include "engine/kernels/all_reduce.h"
#include "engine/kernels/attention.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/kernels/rope.h"
#include "engine/kernels/swiglu.h"
#include "engine/runtime/sharded_runner.h"

namespace engine {

void sharded_layer(const Blas& blas, const ModelConfig& shard, std::span<const RankLayer> ranks,
                   const Tensor& x, const Tensor& norm, const Tensor& positions, int64_t start) {
    const Stream stream = blas.stream();
    const auto eps = float(shard.rms_eps);
    std::vector<Tensor> partials;
    for (const RankLayer& r : ranks) partials.push_back(r.partial);

    naive::rmsnorm(norm, x, ranks.front().weights.ln1, eps, stream);
    for (const RankLayer& r : ranks) {
        gemm(blas, r.qkv, norm, r.weights.wqkv);
        naive::rope(r.qkv, positions, shard.heads, shard.kv_heads, float(shard.rope_theta), stream);
        naive::store_kv(r.k_cache, r.v_cache, r.qkv, start, stream);
        naive::attention(r.attn, r.qkv, r.k_cache, r.v_cache, start, shard.heads, shard.kv_heads,
                         stream);
        gemm(blas, r.partial, r.attn, r.weights.wo);
    }
    all_reduce_add(x, partials, stream);

    naive::rmsnorm(norm, x, ranks.front().weights.ln2, eps, stream);
    for (const RankLayer& r : ranks) {
        gemm(blas, r.gate_up, norm, r.weights.wgu);
        naive::swiglu(r.act, r.gate_up, stream);
        gemm(blas, r.partial, r.act, r.weights.wdown);
    }
    all_reduce_add(x, partials, stream);
}

}  // namespace engine
