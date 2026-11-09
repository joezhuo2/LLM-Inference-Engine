#include "engine/sched/graph_buckets.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace engine {
std::vector<int64_t> decode_graph_buckets(int64_t max_batch, int64_t steps_per_octave) {
    if (max_batch < 1) {
        throw std::invalid_argument("max_batch must be at least 1");
    }

    if (steps_per_octave < 1 || (steps_per_octave & (steps_per_octave - 1)) != 0) {
        throw std::invalid_argument("steps_per_octave must be at least 1 and a power of 2");
    }

    std::vector<int64_t> buckets;
    int64_t octave_start = 1;

    while (octave_start < max_batch) {
        int64_t step = std::max<int64_t>(1, octave_start / steps_per_octave);
        int64_t next_octave = octave_start * 2;

        for (int64_t val = octave_start; val < next_octave && val < max_batch; val += step) {
            buckets.push_back(val);
        }

        octave_start = next_octave;
    }

    if (buckets.empty() || buckets.back() != max_batch) {
        buckets.push_back(max_batch);
    }

    return buckets;
}

}  // namespace engine
