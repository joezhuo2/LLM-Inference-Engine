#pragma once

#include <cstdint>
#include <deque>
#include <span>
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
};

struct Step {
    bool prefill = false;
    std::vector<SeqId> seqs{};
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);

    void submit(Request request);
    Step schedule();
    void update(std::span<const int32_t> tokens);
    std::vector<Request> take_finished();

    const Request& request(SeqId id) const;
    const BlockManager& blocks() const;
    int num_waiting() const;
    int num_running() const;

private:
    SchedulerConfig config_;
    BlockManager block_manager_;
    std::deque<SeqId> waiting_queue_;
    std::vector<SeqId> running_list_;
    std::vector<Request> finished_list_;
    std::unordered_map<SeqId, Request> requests_;
    bool pending_step_ = false;
    Step pending_step_obj_;
};

}  // namespace engine
