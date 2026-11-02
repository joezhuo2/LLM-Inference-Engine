#pragma once

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "engine/server/backend.h"
#include "engine/server/broadcaster.h"
#include "engine/server/channel.h"
#include "engine/server/chat_api.h"
#include "engine/server/metrics_window.h"
#include "engine/server/stop_matcher.h"
#include "engine/tokenizer/text_stream.h"

namespace engine::server {

struct Submission {
    std::vector<int32_t> prompt{};
    int max_tokens = 1;
    SamplingParams params{};
    std::vector<std::string> stop{};
};

struct TokenEvent {
    std::string text{};
    int64_t step = 0;
    std::optional<FinishReason> finish{};
    int completion_tokens = 0;
};

using Events = Channel<TokenEvent>;

struct Ticket {
    SeqId id = 0;
    std::shared_ptr<Events> events{};
};

struct Counters {
    int waiting = 0;
    int running = 0;
    int blocks_free = 0;
    int64_t finished_total = 0;
    int64_t preemptions_total = 0;
    int64_t output_total = 0;
    int64_t steps_total = 0;
};

struct MetricsSnapshot {
    double time_s = 0.0;
    Counters counters{};
    int blocks_total = 0;
    double output_per_s = 0.0;
    std::optional<Percentiles> ttft_s{}, tpot_s{};
};

class ServingLoop {
public:
    ServingLoop(Backend& backend, Decode decode, int32_t eos_id);
    ~ServingLoop();
    ServingLoop(const ServingLoop&) = delete;
    ServingLoop& operator=(const ServingLoop&) = delete;

    Ticket submit(Submission submission);
    void cancel(SeqId id);

    bool run_once();
    void run(std::stop_token stop);

    MetricsSnapshot metrics();
    StepBroadcaster& steps() { return steps_; }
    Clock::time_point start() const { return start_; }

private:
    struct Pending {
        SeqId id;
        Submission submission;
        std::shared_ptr<Events> events;
        Clock::time_point arrived;
    };

    struct Live {
        std::shared_ptr<Events> events;
        TextStream text;
        StopMatcher stop;
        Clock::time_point arrived{}, first_token{};
        int generated = 0;
        int delivered = 0;
        bool done = false;
    };

    Backend& backend_;
    Decode decode_;
    int32_t eos_id_;
    Clock::time_point start_;

    std::mutex inbox_mutex_;
    std::condition_variable inbox_ready_;
    std::vector<Pending> submissions_;
    std::vector<SeqId> cancels_;
    SeqId next_id_ = 0;

    std::map<SeqId, Live> live_;
    bool warned_cancel_ = false;

    std::mutex metrics_mutex_;
    Counters counters_;
    MetricsWindow window_;

    StepBroadcaster steps_;

    double seconds(Clock::time_point t) const;
    void take_inbox();
    void admit(Pending pending);
    void drop(SeqId id);
    void release(SeqId id);
    void handle(const StepResult& result);
    std::vector<std::string> step_texts(const StepTrace& trace);
    bool deliver(Live& live, const std::string& text, int64_t step, Clock::time_point when);
    void finish(Live& live, FinishReason reason, int64_t step, Clock::time_point when);
    void record_step(const StepTrace& trace, int preemptions);
    void record_queues();
};

}  // namespace engine::server
