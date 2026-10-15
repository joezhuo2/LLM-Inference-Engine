#pragma once

#include <cstdint>
#include <vector>

#include "engine/sampling/sample.h"
#include "engine/sched/scheduler.h"

namespace engine {

struct PagedBatch {
    bool prefill = false;
    int64_t num_seqs = 0;
    int64_t max_blocks = 0;
    std::vector<int32_t> token_ids{};
    std::vector<int32_t> positions{};
    std::vector<int32_t> slot_mapping{};
    std::vector<int32_t> context_lens{};
    std::vector<int32_t> logit_rows{};
    std::vector<int32_t> block_tables{};

    int64_t num_tokens() const { return int64_t(token_ids.size()); }
};

PagedBatch build_paged_batch(const Step& step, const Scheduler& scheduler, int max_blocks);

std::vector<SampleRow> sample_rows(const Step& step, const Scheduler& scheduler);

}  // namespace engine
