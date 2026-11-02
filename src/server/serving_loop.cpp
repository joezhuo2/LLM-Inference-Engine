#include "engine/server/serving_loop.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace engine::server {

namespace {

constexpr auto kIdleWait = std::chrono::milliseconds(100);

}  // namespace

ServingLoop::ServingLoop(Backend& backend, Decode decode, int32_t eos_id)
    : backend_(backend), decode_(std::move(decode)), eos_id_(eos_id), start_(Clock::now()) {
    record_queues();
}

ServingLoop::~ServingLoop() {
    steps_.close();
    for (auto& [id, live] : live_) live.events->close();
    for (Pending& pending : submissions_) pending.events->close();
}

Ticket ServingLoop::submit(Submission submission) {
    auto events = std::make_shared<Events>();
    SeqId id = 0;
    {
        const std::lock_guard lock(inbox_mutex_);
        id = next_id_++;
        submissions_.push_back({id, std::move(submission), events, Clock::now()});
    }
    inbox_ready_.notify_one();
    return {id, std::move(events)};
}

void ServingLoop::cancel(SeqId id) {
    {
        const std::lock_guard lock(inbox_mutex_);
        cancels_.push_back(id);
    }
    inbox_ready_.notify_one();
}

bool ServingLoop::run_once() {
    take_inbox();
    Scheduler& scheduler = backend_.scheduler();
    if (scheduler.num_waiting() + scheduler.num_running() == 0) {
        record_queues();
        return false;
    }
    handle(backend_.step());
    record_queues();
    return true;
}

void ServingLoop::run(std::stop_token stop) {
    while (!stop.stop_requested()) {
        if (run_once()) continue;
        std::unique_lock lock(inbox_mutex_);
        inbox_ready_.wait_for(lock, kIdleWait,
                              [&] { return !submissions_.empty() || !cancels_.empty(); });
    }
}

MetricsSnapshot ServingLoop::metrics() {
    const std::lock_guard lock(metrics_mutex_);
    const double now = seconds(Clock::now());
    return {.time_s = now,
            .counters = counters_,
            .blocks_total = backend_.num_blocks(),
            .output_per_s = window_.output_per_s(now),
            .ttft_s = window_.ttft_s(now),
            .tpot_s = window_.tpot_s(now)};
}

double ServingLoop::seconds(Clock::time_point t) const {
    return std::chrono::duration<double>(t - start_).count();
}

void ServingLoop::take_inbox() {
    std::vector<Pending> submissions;
    std::vector<SeqId> cancels;
    {
        const std::lock_guard lock(inbox_mutex_);
        submissions.swap(submissions_);
        cancels.swap(cancels_);
    }
    for (Pending& pending : submissions) admit(std::move(pending));
    for (const SeqId id : cancels) drop(id);
}

void ServingLoop::admit(Pending pending) {
    Request request{.id = pending.id,
                    .prompt = std::move(pending.submission.prompt),
                    .max_new_tokens = pending.submission.max_tokens,
                    .params = pending.submission.params};
    try {
        backend_.scheduler().submit(std::move(request));
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        pending.events->close();
        return;
    }
    live_.emplace(pending.id, Live{.events = std::move(pending.events),
                                   .text = TextStream(decode_),
                                   .stop = StopMatcher(std::move(pending.submission.stop)),
                                   .arrived = pending.arrived});
}

void ServingLoop::drop(SeqId id) {
    const auto it = live_.find(id);
    if (it == live_.end() || it->second.done) return;
    it->second.done = true;
    it->second.events->close();
    release(id);
}

void ServingLoop::release(SeqId id) {
    try {
        backend_.scheduler().cancel(id);
        live_.erase(id);
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::logic_error& e) {
        if (!warned_cancel_)
            std::fprintf(stderr,
                         "warning: %s; a cancelled request keeps decoding, with its tokens "
                         "discarded, and holds its KV blocks until it reaches max_tokens or EOS\n",
                         e.what());
        warned_cancel_ = true;
    }
}

void ServingLoop::handle(const StepResult& result) {
    const StepTrace& trace = result.trace;
    const int64_t index = counters_.steps_total;
    const std::vector<std::string> texts = step_texts(trace);
    if (steps_.has_subscribers()) steps_.publish(step_json(trace, index, start_, texts).dump());

    int preemptions = 0;
    std::vector<SeqId> stopped;
    for (size_t j = 0; j < trace.seqs.size(); ++j) {
        Live& live = live_.at(trace.seqs[j]);
        if (trace.prefill && live.generated > 0) ++preemptions;
        ++live.generated;
        if (!live.done && deliver(live, texts[j], index, trace.end))
            stopped.push_back(trace.seqs[j]);
    }
    for (const Request& request : result.finished) {
        Live& live = live_.at(request.id);
        if (!live.done) {
            const std::string tail = live.stop.flush();
            if (!tail.empty()) live.events->push({.text = tail, .step = index});
            const bool eos = !request.output.empty() && request.output.back() == eos_id_;
            finish(live, eos ? FinishReason::Stop : FinishReason::Length, index, trace.end);
        }
        live_.erase(request.id);
    }
    for (const SeqId id : stopped)
        if (live_.count(id)) release(id);
    record_step(trace, preemptions);
}

std::vector<std::string> ServingLoop::step_texts(const StepTrace& trace) {
    std::vector<std::string> texts;
    for (size_t j = 0; j < trace.seqs.size(); ++j) {
        const int32_t token = trace.tokens[j];
        Live& live = live_.at(trace.seqs[j]);
        texts.push_back(token == eos_id_ ? std::string() : live.text.push(token));
    }
    return texts;
}

bool ServingLoop::deliver(Live& live, const std::string& text, int64_t step,
                          Clock::time_point when) {
    if (live.delivered == 0) live.first_token = when;
    ++live.delivered;
    const StopMatcher::Result out = live.stop.push(text);
    if (!out.text.empty()) live.events->push({.text = out.text, .step = step});
    if (out.stopped) finish(live, FinishReason::Stop, step, when);
    return out.stopped;
}

void ServingLoop::finish(Live& live, FinishReason reason, int64_t step, Clock::time_point when) {
    live.events->push({.step = step, .finish = reason, .completion_tokens = live.delivered});
    live.events->close();
    live.done = true;
    const double ttft = std::chrono::duration<double>(live.first_token - live.arrived).count();
    std::optional<double> tpot;
    if (live.delivered > 1)
        tpot = std::chrono::duration<double>(when - live.first_token).count() /
               double(live.delivered - 1);
    const std::lock_guard lock(metrics_mutex_);
    ++counters_.finished_total;
    window_.add_request(seconds(when), ttft, tpot);
}

void ServingLoop::record_step(const StepTrace& trace, int preemptions) {
    const std::lock_guard lock(metrics_mutex_);
    ++counters_.steps_total;
    counters_.output_total += int64_t(trace.seqs.size());
    counters_.preemptions_total += preemptions;
    window_.add_tokens(seconds(trace.end), int64_t(trace.seqs.size()));
}

void ServingLoop::record_queues() {
    const Scheduler& scheduler = backend_.scheduler();
    const std::lock_guard lock(metrics_mutex_);
    counters_.waiting = scheduler.num_waiting();
    counters_.running = scheduler.num_running();
    counters_.blocks_free = scheduler.blocks().num_free();
}

}  // namespace engine::server
