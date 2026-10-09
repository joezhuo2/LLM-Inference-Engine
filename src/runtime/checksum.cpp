#include "engine/runtime/checksum.h"

#include <algorithm>
#include <cstddef>

#include "kernels/cuda_check.h"

namespace engine {

std::vector<TensorChecksum> checksum_device(const WeightLayout& layout, const void* device) {
    std::vector<CopyOp> copies = layout.copies;
    std::sort(copies.begin(), copies.end(),
              [](const CopyOp& a, const CopyOp& b) { return a.name < b.name; });
    uint64_t largest = 0;
    for (const CopyOp& c : copies) largest = std::max(largest, c.bytes);

    std::vector<std::byte> host(largest);
    std::vector<TensorChecksum> out;
    for (const CopyOp& c : copies) {
        CUDA_CHECK(cudaMemcpy(host.data(), static_cast<const std::byte*>(device) + c.dst, c.bytes,
                              cudaMemcpyDeviceToHost));
        out.push_back(checksum_bf16(c.name, c.shape, {host.data(), c.bytes}));
    }
    return out;
}

}  // namespace engine
