#pragma once

#include <cstdint>
#include <deque>
#include <span>
#include <unordered_map>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/kv/cache_size.h"
#include "engine/sched/request.h"

namespace engine {

struct SchedulerConfig {
    int num_blocks = 0;
    int block_size = kKvBlockSize;
    int max_batch = 16;
    int max_prefill_tokens = 2048;
    int max_context = 2048;
    int32_t eos_id = -1;
    int max_draft = 0;
    int max_ngram = 3;
};

struct Step {
    bool prefill = false;
    std::vector<SeqId> seqs{};
    std::vector<std::vector<int32_t>> drafts{};
};

struct PrefillChunk {
    SeqId id = 0;
    int first = 0;
    int count = 0;
};

struct ChunkedStep {
    std::vector<SeqId> decode{};
    std::vector<PrefillChunk> prefill{};
    std::vector<SeqId> sampled{};
    std::vector<std::vector<int32_t>> drafts{};

    bool empty() const { return decode.empty() && prefill.empty(); }
    int num_tokens() const {
        int tokens = int(decode.size());
        for (const PrefillChunk& chunk : prefill) tokens += chunk.count;
        for (const std::vector<int32_t>& draft : drafts) tokens += int(draft.size());
        return tokens;
    }
};

struct DraftCounts {
    int64_t proposed = 0;
    int64_t accepted = 0;
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);

    void submit(Request request);
    Step schedule();
    ChunkedStep schedule_chunked(int token_budget);
    void update(std::span<const int32_t> tokens);
    std::vector<Request> take_finished();
    void cancel(SeqId id);

    const Request& request(SeqId id) const;
    const BlockManager& blocks() const;
    int num_waiting() const;
    int num_running() const;
    const DraftCounts& draft_counts() const;

private:
    std::vector<std::vector<int32_t>> reserve_drafts(std::span<const SeqId> decode, int spare_rows);

    SchedulerConfig config_;
    BlockManager block_manager_;
    std::deque<SeqId> waiting_queue_;
    std::vector<SeqId> running_list_;
    std::vector<Request> finished_list_;
    std::unordered_map<SeqId, Request> requests_;
    bool pending_step_ = false;
    Step pending_step_obj_;
    std::unordered_map<SeqId, int> prefill_progress_;
    std::vector<SeqId> pending_sampled_seqs_;
    std::vector<std::vector<int32_t>> pending_drafts_;
    DraftCounts draft_counts_;
};

}  // namespace engine
