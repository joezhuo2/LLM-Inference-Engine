#include "engine/server/gpu_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/kv/cache_size.h"
#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/runtime/serve.h"

namespace engine::server {

namespace {

constexpr int kMaxPrefillTokens = 2048;

struct OwnedStream {
    cudaStream_t stream = nullptr;

    explicit OwnedStream(bool create) {
        if (create && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("cannot create a CUDA stream");
    }
    ~OwnedStream() {
        if (stream) cudaStreamDestroy(stream);
    }
    OwnedStream(const OwnedStream&) = delete;
    OwnedStream& operator=(const OwnedStream&) = delete;
};

std::vector<int64_t> graph_buckets(int max_batch) {
    std::vector<int64_t> buckets;
    for (int64_t b = 1; b < max_batch; b *= 2) buckets.push_back(b);
    buckets.push_back(max_batch);
    return buckets;
}

SchedulerConfig scheduler_config(const ModelConfig& config, int num_blocks, int max_batch) {
    return {.num_blocks = num_blocks,
            .block_size = kKvBlockSize,
            .max_batch = max_batch,
            .max_prefill_tokens = std::max(kMaxPrefillTokens, int(config.max_positions)),
            .max_context = int(config.max_positions),
            .eos_id = config.eos_id};
}

class GpuBackend final : public Backend {
public:
    GpuBackend(const DeviceWeights& w, const GpuBackendOptions& o)
        : config_(scheduler_config(
              w.config, kv_cache_blocks(w.config, kKvBlockSize, device_free_bytes()), o.max_batch)),
          scheduler_(config_),
          stream_(o.graphs),
          cache_(w.config, config_.num_blocks, kKvBlockSize),
          runner_(w.config, w.weights, cache_, config_.max_prefill_tokens, o.max_batch,
                  config_.max_context, stream_.stream),
          sampler_(o.max_batch),
          next_(size_t(o.max_batch) * sizeof(int32_t)) {
        if (o.graphs) graphs_.emplace(runner_, graph_buckets(o.max_batch));
    }

    Scheduler& scheduler() override { return scheduler_; }
    int num_blocks() const override { return config_.num_blocks; }

    StepResult step() override {
        std::vector<StepTrace> trace;
        StepResult result;
        result.finished = serve_step(scheduler_, runner_, sampler_, next_, nullptr,
                                     graphs_ ? &*graphs_ : nullptr, &trace);
        result.trace = std::move(trace.back());
        return result;
    }

private:
    SchedulerConfig config_;
    Scheduler scheduler_;
    OwnedStream stream_;
    KvCache cache_;
    PagedRunner runner_;
    Sampler sampler_;
    DeviceBuffer next_;
    std::optional<DecodeGraphs> graphs_;
};

}  // namespace

std::unique_ptr<Backend> make_gpu_backend(const DeviceWeights& weights,
                                          const GpuBackendOptions& options) {
    if (options.max_batch < 1) throw std::invalid_argument("serve: --max-batch must be at least 1");
    return std::make_unique<GpuBackend>(weights, options);
}

std::string gpu_name() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return "unknown";
    return prop.name;
}

}  // namespace engine::server
