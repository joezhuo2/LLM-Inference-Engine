#include "engine/loader/tensor_parallel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

constexpr uint64_t align_up(uint64_t byte_os, uint64_t alignment) {
    return (byte_os + alignment - 1) & ~(alignment - 1);
}

WeightLayout plan_shard_layout(const ModelConfig&, const SafetensorsHeader&, int, int) {
    throw std::logic_error("unimplemented: plan_shard_layout");
}

}  // namespace engine
