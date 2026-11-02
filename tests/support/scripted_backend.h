#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <utility>

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
                             std::chrono::milliseconds delay = std::chrono::milliseconds(0))
        : scheduler_({.num_blocks = num_blocks,
                      .block_size = block_size,
                      .max_batch = 8,
                      .max_prefill_tokens = max_context,
                      .max_context = max_context,
                      .eos_id = kEos}),
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
        for (const SeqId seq : step.seqs) r.trace.tokens.push_back(next_(scheduler_.request(seq)));
        std::this_thread::sleep_for(delay_);
        scheduler_.update(r.trace.tokens);
        r.finished = scheduler_.take_finished();
        for (const Request& done : r.finished) {
            r.trace.finished.push_back(done.id);
            preemptions_ += done.preemptions;
        }
        r.trace.prefill = step.prefill;
        r.trace.seqs = step.seqs;
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
