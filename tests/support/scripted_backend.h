#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine/server/backend.h"

namespace engine::test {

inline constexpr int32_t kEos = 2;

inline std::string word_decode(std::span<const int32_t> ids) {
    std::string out;
    for (const int32_t id : ids)
        if (id != kEos) out += " w" + std::to_string(id);
    return out;
}

class ScriptedBackend : public server::Backend {
public:
    using Next = std::function<int32_t(const Request&)>;

    explicit ScriptedBackend(Next next, int num_blocks = 64, int block_size = 16,
                             int max_context = 256,
                             std::chrono::milliseconds delay = std::chrono::milliseconds(0),
                             int max_draft = 0)
        : scheduler_({.num_blocks = num_blocks,
                      .block_size = block_size,
                      .max_batch = 8,
                      .max_prefill_tokens = max_context,
                      .max_context = max_context,
                      .eos_id = kEos,
                      .max_draft = max_draft}),
          num_blocks_(num_blocks),
          next_(std::move(next)),
          delay_(delay) {}

    Scheduler& scheduler() override { return scheduler_; }
    int num_blocks() const override { return num_blocks_; }
    int preemptions() const { return preemptions_; }

    server::StepResult step() override {
        server::StepResult r;
        r.trace.start = Clock::now();
        const Step step = scheduler_.schedule();
        std::vector<int32_t> sampled;
        std::vector<size_t> before;
        for (size_t i = 0; i < step.seqs.size(); ++i) {
            Request row = scheduler_.request(step.seqs[i]);
            before.push_back(row.output.size());
            sampled.push_back(next_(row));
            if (i >= step.drafts.size()) continue;
            for (const int32_t guess : step.drafts[i]) {
                row.output.push_back(guess);
                sampled.push_back(next_(row));
            }
        }
        const DraftCounts counts = scheduler_.draft_counts();
        std::this_thread::sleep_for(delay_);
        scheduler_.update(sampled);
        r.finished = scheduler_.take_finished();
        for (const Request& done : r.finished) {
            r.trace.finished.push_back(done.id);
            preemptions_ += done.preemptions;
        }
        for (size_t i = 0; i < step.seqs.size(); ++i) {
            const SeqId seq = step.seqs[i];
            const auto done = std::find_if(r.finished.begin(), r.finished.end(),
                                           [seq](const Request& f) { return f.id == seq; });
            const Request& now = done != r.finished.end() ? *done : scheduler_.request(seq);
            for (size_t j = before[i]; j < now.output.size(); ++j) {
                r.trace.seqs.push_back(seq);
                r.trace.tokens.push_back(now.output[j]);
            }
        }
        r.trace.drafted = int(scheduler_.draft_counts().proposed - counts.proposed);
        r.trace.accepted = int(scheduler_.draft_counts().accepted - counts.accepted);
        r.trace.prefill = step.prefill;
        r.trace.waiting = scheduler_.num_waiting();
        r.trace.running = scheduler_.num_running();
        r.trace.free_blocks = scheduler_.blocks().num_free();
        r.trace.end = Clock::now();
        return r;
    }

private:
    Scheduler scheduler_;
    int num_blocks_;
    Next next_;
    std::chrono::milliseconds delay_;
    int preemptions_ = 0;
};

}  // namespace engine::test
