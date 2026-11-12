#include "engine/sched/paged_batch.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine {

namespace {

constexpr int64_t kMaxSplits = 8;
constexpr int64_t kMinSplitTokens = 256;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("[build_paged_batch] " + what);
}

std::span<const int32_t> draft_of(std::span<const std::vector<int32_t>> drafts, size_t i) {
    return i < drafts.size() ? std::span<const int32_t>(drafts[i]) : std::span<const int32_t>();
}

bool append_rows(PagedBatch& batch, const Scheduler& scheduler, SeqId id, int first, int count,
                 std::span<const int32_t> draft = {}) {
    const BlockManager& blocks = scheduler.blocks();
    const Request& request = scheduler.request(id);
    const int length = blocks.length(id);
    const std::string name = "sequence " + std::to_string(id);
    require(request.state == RequestState::Running, name + " is not running");
    const int64_t known = request.length();
    require(length == known + int64_t(draft.size()),
            name + " holds a different number of slots than tokens");
    const std::vector<int>& table = blocks.table(id);
    require(int64_t(table.size()) <= batch.max_blocks, name + " needs more blocks than max_blocks");
    require(first >= 0 && count > 0 && first + count <= length, name + " has rows past its tokens");

    for (int pos = first; pos < first + count; ++pos) {
        batch.token_ids.push_back(pos < known ? request.token(pos) : draft[size_t(pos - known)]);
        batch.positions.push_back(pos);
        batch.slot_mapping.push_back(blocks.slot(id, pos));
        batch.context_lens.push_back(pos + 1);
        const size_t row = batch.block_tables.size();
        batch.block_tables.resize(row + size_t(batch.max_blocks), 0);
        std::copy(table.begin(), table.end(), batch.block_tables.begin() + int64_t(row));
    }
    const bool last = first + count == length;
    if (last) {
        const size_t rows = batch.token_ids.size();
        for (size_t row = rows - 1 - draft.size(); row < rows; ++row)
            batch.logit_rows.push_back(int32_t(row));
    }
    return last;
}

}  // namespace

PagedBatch build_paged_batch(const Step& step, const Scheduler& scheduler, int max_blocks) {
    require(max_blocks > 0, "max_blocks must be positive");
    require(!step.seqs.empty(), "the step has no sequences");

    PagedBatch batch;
    batch.prefill = step.prefill;
    batch.num_seqs = int64_t(step.seqs.size());
    batch.max_blocks = max_blocks;
    require(step.drafts.empty() || (!step.prefill && step.drafts.size() == step.seqs.size()),
            "drafts must be empty or hold one entry per decode sequence");
    for (size_t i = 0; i < step.seqs.size(); ++i) {
        const SeqId id = step.seqs[i];
        const std::span<const int32_t> draft = draft_of(step.drafts, i);
        const int length = scheduler.blocks().length(id);
        const int first = step.prefill ? 0 : length - 1 - int(draft.size());
        if (step.prefill) {
            batch.chunk_starts.push_back(int32_t(batch.num_tokens()));
            batch.cached_lens.push_back(0);
        }
        append_rows(batch, scheduler, id, first, length - first, draft);
    }
    if (step.prefill) batch.chunk_starts.push_back(int32_t(batch.num_tokens()));
    batch.num_seqs = int64_t(batch.logit_rows.size());
    return batch;
}

PagedBatch build_chunked_batch(const ChunkedStep& step, const Scheduler& scheduler,
                               int max_blocks) {
    require(max_blocks > 0, "max_blocks must be positive");
    require(!step.empty(), "the step has no rows");

    PagedBatch batch;
    batch.prefill = !step.prefill.empty();
    batch.num_seqs = int64_t(step.sampled.size());
    batch.max_blocks = max_blocks;
    require(step.drafts.empty() || step.drafts.size() == step.decode.size(),
            "drafts must be empty or hold one entry per decode sequence");
    std::vector<SeqId> ends;
    for (size_t i = 0; i < step.decode.size(); ++i) {
        const SeqId id = step.decode[i];
        const std::span<const int32_t> draft = draft_of(step.drafts, i);
        const int rows = 1 + int(draft.size());
        if (append_rows(batch, scheduler, id, scheduler.blocks().length(id) - rows, rows, draft))
            ends.push_back(id);
    }
    for (const PrefillChunk& chunk : step.prefill) {
        batch.chunk_starts.push_back(int32_t(batch.num_tokens()));
        batch.cached_lens.push_back(chunk.first);
        if (append_rows(batch, scheduler, chunk.id, chunk.first, chunk.count))
            ends.push_back(chunk.id);
    }
    if (batch.prefill) batch.chunk_starts.push_back(int32_t(batch.num_tokens()));
    require(ends == step.sampled,
            "sampled must list the sequences whose rows reach their last token, in row order");
    batch.num_seqs = int64_t(batch.logit_rows.size());
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

int64_t PagedBatch::longest_context() const {
    return context_lens.empty() ? 0 : *std::max_element(context_lens.begin(), context_lens.end());
}

int64_t PagedBatch::longest_decode_context() const {
    const auto end = context_lens.begin() + decode_rows();
    return end == context_lens.begin() ? 0 : *std::max_element(context_lens.begin(), end);
}

int64_t attention_splits(int64_t rows, int64_t heads, int64_t longest_context) {
    int64_t splits = 1;
    while (splits < kMaxSplits && 2 * splits * rows * heads <= kSplitAttentionBlocks &&
           longest_context >= 2 * splits * kMinSplitTokens)
        splits *= 2;
    return splits;
}

std::vector<SampleRow> sample_rows(std::span<const SeqId> seqs, const Scheduler& scheduler,
                                   std::span<const std::vector<int32_t>> drafts) {
    std::vector<SampleRow> rows;
    rows.reserve(seqs.size());
    for (size_t i = 0; i < seqs.size(); ++i) {
        const Request& request = scheduler.request(seqs[i]);
        const uint64_t step = request.output.size();
        for (uint64_t j = 0; j <= draft_of(drafts, i).size(); ++j)
            rows.push_back(SampleRow{request.params, step + j});
    }
    return rows;
}

}  // namespace engine
