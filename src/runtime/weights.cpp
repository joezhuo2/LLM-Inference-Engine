#include "engine/runtime/weights.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"
#include "engine/runtime/pinned_buffer.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

struct Event {
    cudaEvent_t event;

    Event() { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    ~Event() { CUDA_CHECK(cudaEventDestroy(event)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

}  // namespace

UploadStats upload_weights(std::span<const std::byte> file, const WeightLayout& layout,
                           void* device, Stream stream, size_t chunk_bytes) {
    if (chunk_bytes == 0)
        throw std::invalid_argument("upload_weights: chunk_bytes must be positive");
    std::array<PinnedBuffer, 2> staging{PinnedBuffer(chunk_bytes), PinnedBuffer(chunk_bytes)};
    std::array<Event, 2> drained;
    auto* dst = static_cast<std::byte*>(device);

    UploadStats stats;
    const auto start = std::chrono::steady_clock::now();
    size_t chunk = 0;
    for (const CopyOp& c : layout.copies) {
        for (uint64_t off = 0; off < c.bytes; off += chunk_bytes, ++chunk) {
            const size_t n = size_t(std::min<uint64_t>(chunk_bytes, c.bytes - off));
            PinnedBuffer& buf = staging[chunk % 2];
            cudaEvent_t ready = drained[chunk % 2].event;
            CUDA_CHECK(cudaEventSynchronize(ready));
            std::memcpy(buf.data(), file.data() + c.src + off, n);
            CUDA_CHECK(
                cudaMemcpyAsync(dst + c.dst + off, buf.data(), n, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(ready, stream));
        }
        stats.bytes += c.bytes;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}

DeviceWeights load_weights(const std::filesystem::path& model_dir, Stream stream) {
    DeviceWeights w;
    w.config = load_config(model_dir / "config.json");
    const MappedFile file(model_dir / "model.safetensors");
    w.layout = plan_weight_layout(w.config, parse_safetensors_header(file.bytes()));
    w.buffer = DeviceBuffer(w.layout.total_bytes);
    w.stats = upload_weights(file.bytes(), w.layout, w.buffer.data(), stream);
    w.weights = bind_weights(w.layout, w.buffer.data());
    return w;
}

}  // namespace engine
