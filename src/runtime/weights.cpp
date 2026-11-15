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
    size_t staged = 0;
    uint64_t staged_dst = 0;
    const auto flush = [&] {
        if (staged == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(dst + staged_dst, staging[chunk % 2].data(), staged,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(drained[chunk % 2].event, stream));
        ++chunk;
        staged = 0;
    };
    const CopyOp* previous = nullptr;
    for (const CopyOp& c : layout.copies) {
        if (previous && previous->name != c.name) flush();
        previous = &c;
        for (uint64_t off = 0; off < c.bytes;) {
            if (staged == chunk_bytes || staged_dst + staged != c.dst + off) flush();
            if (staged == 0) {
                CUDA_CHECK(cudaEventSynchronize(drained[chunk % 2].event));
                staged_dst = c.dst + off;
            }
            const size_t n = size_t(std::min<uint64_t>(chunk_bytes - staged, c.bytes - off));
            std::memcpy(static_cast<std::byte*>(staging[chunk % 2].data()) + staged,
                        file.data() + c.src + off, n);
            staged += n;
            off += n;
        }
        stats.bytes += c.bytes;
    }
    flush();
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
