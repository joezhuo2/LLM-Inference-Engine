#include "engine/sched/speculative.h"

#include <algorithm>
#include <string>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/kv/block_manager.h"

namespace engine {

std::vector<int32_t> propose_prompt_lookup(std::span<const int32_t> context, int max_ngram,
                                           int max_draft) {
    if (max_ngram < 1 || max_draft < 1) {
        throw std::invalid_argument(
            "[propose_prompt_lookup] max_ngram and max_draft must be at least 1");
    }

    const size_t n = context.size();
    if (n < 2) return {};

    const size_t max_g = std::min(static_cast<size_t>(max_ngram), n - 1);

    for (size_t g = max_g; g >= 1; --g) {
        const size_t pattern_start = n - g;
        const size_t max_i = n - 1 - g;

        for (size_t k = max_i + 1; k >= 1; --k) {
            const size_t i = k - 1;
            if (std::equal(context.begin() + i, context.begin() + i + g,
                           context.begin() + pattern_start)) {
                const size_t draft_start = i + g;
                const size_t draft_end = std::min(draft_start + static_cast<size_t>(max_draft), n);

                return std::vector<int32_t>(context.begin() + draft_start,
                                            context.begin() + draft_end);
            }
        }
    }

    return {};
}

int count_accepted(std::span<const int32_t>, std::span<const int32_t>) {
    throw std::logic_error("unimplemented: count_accepted");
}

void BlockManager::truncate(SeqId, int) {
    throw std::logic_error("unimplemented: BlockManager::truncate");
}

}  // namespace engine
