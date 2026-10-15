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

PagedBatch pad_decode_batch(const PagedBatch& batch, int64_t rows) {
    const auto reject = [](const std::string& what) {
        throw std::invalid_argument("[pad_decode_batch] " + what);
    };
    if (batch.prefill) reject("the batch is a prefill batch");
    if (batch.num_seqs < 1 || batch.num_tokens() != batch.num_seqs)
        reject("a decode batch has one row per sequence");
    if (rows < batch.num_seqs) reject("rows must be at least the number of sequences");
    for (int64_t i = 0; i < batch.num_seqs; ++i)
        if (batch.logit_rows[size_t(i)] != i) reject("logit_rows must be the identity");

    PagedBatch padded = batch;
    padded.num_seqs = rows;
    padded.token_ids.resize(size_t(rows), 0);
    padded.positions.resize(size_t(rows), 0);
    padded.slot_mapping.resize(size_t(rows), -1);
    padded.context_lens.resize(size_t(rows), 0);
    padded.block_tables.resize(size_t(rows * batch.max_blocks), 0);
    padded.logit_rows.resize(size_t(rows));
    for (int64_t i = batch.num_seqs; i < rows; ++i) padded.logit_rows[size_t(i)] = int32_t(i);
    return padded;
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
