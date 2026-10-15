#include "engine/sched/paged_batch.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine {

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("[build_paged_batch] " + what);
}

}  // namespace

PagedBatch build_paged_batch(const Step& step, const Scheduler& scheduler, int max_blocks) {
    require(max_blocks > 0, "max_blocks must be positive");
    require(!step.seqs.empty(), "the step has no sequences");

    const BlockManager& blocks = scheduler.blocks();
    PagedBatch batch;
    batch.prefill = step.prefill;
    batch.num_seqs = int64_t(step.seqs.size());
    batch.max_blocks = max_blocks;

    for (const SeqId id : step.seqs) {
        const Request& request = scheduler.request(id);
        const int length = blocks.length(id);
        require(request.state == RequestState::Running,
                "sequence " + std::to_string(id) + " is not running");
        require(length == request.length(), "sequence " + std::to_string(id) +
                                                " holds a different number of slots than tokens");
        const std::vector<int>& table = blocks.table(id);
        require(int64_t(table.size()) <= max_blocks,
                "sequence " + std::to_string(id) + " needs more blocks than max_blocks");

        const int first = step.prefill ? 0 : length - 1;
        for (int pos = first; pos < length; ++pos) {
            batch.token_ids.push_back(request.token(pos));
            batch.positions.push_back(pos);
            batch.slot_mapping.push_back(blocks.slot(id, pos));
            batch.context_lens.push_back(pos + 1);
            const size_t row = batch.block_tables.size();
            batch.block_tables.resize(row + size_t(max_blocks), 0);
            std::copy(table.begin(), table.end(), batch.block_tables.begin() + int64_t(row));
        }
        batch.logit_rows.push_back(int32_t(batch.token_ids.size() - 1));
    }
    return batch;
}

std::vector<SampleRow> sample_rows(const Step& step, const Scheduler& scheduler) {
    std::vector<SampleRow> rows;
    rows.reserve(step.seqs.size());
    for (const SeqId id : step.seqs) {
        const Request& request = scheduler.request(id);
        rows.push_back(SampleRow{request.params, uint64_t(request.output.size())});
    }
    return rows;
}

}  // namespace engine
