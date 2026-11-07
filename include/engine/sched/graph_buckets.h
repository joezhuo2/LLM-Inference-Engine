#pragma once

#include <cstdint>
#include <vector>

namespace engine {

std::vector<int64_t> decode_graph_buckets(int64_t max_batch, int64_t steps_per_octave);

}  // namespace engine
