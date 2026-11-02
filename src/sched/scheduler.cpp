#include "engine/sched/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/sched/request.h"

namespace engine {

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config), block_manager_(config.num_blocks, config.block_size) {
    if (config.max_batch < 1) {
        throw std::invalid_argument("[SchedulerConfig] max_batch must be >= 1");
    }

    if (config.max_context < 1) {
        throw std::invalid_argument("[SchedulerConfig] max_context must be >= 1");
    }

    if (config.max_prefill_tokens < config.max_context) {
        throw std::invalid_argument("[SchedulerConfig] max_prefill_tokens must be >= max_context");
    }

    int req_blocks = (config.max_context + config.block_size - 1) / config.block_size;

    if (config.num_blocks - 1 < req_blocks) {
        throw std::invalid_argument(
            "[SchedulerConfig] num_blocks - 1 must be >= ceil(max_context / block_size)");
    }
}

void Scheduler::submit(Request request) {
    validate(request, config_.max_context);

    if (requests_.count(request.id)) {
        throw std::invalid_argument("[Scheduler] request id already held");
    }

    request.state = RequestState::Waiting;
    request.preemptions = 0;
    request.arrived = Clock::now();

    request.first_token = Clock::time_point{};
    request.finished = Clock::time_point{};

    requests_[request.id] = request;
    waiting_queue_.push_back(request.id);
}

Step Scheduler::schedule() {
    if (pending_step_) {
        throw std::invalid_argument("[Scheduler] previous non-empty step has not been updated");
    }

    std::vector<SeqId> admitted_seqs;
    int tokens_admitted = 0;

    while (!waiting_queue_.empty() &&
           running_list_.size() + admitted_seqs.size() < size_t(config_.max_batch)) {
        SeqId front_id = waiting_queue_.front();
        const Request& req = requests_.at(front_id);
        int req_length = static_cast<int>(req.length());

        if (!block_manager_.can_allocate(req_length)) {
            break;
        }

        if (tokens_admitted + req_length > config_.max_prefill_tokens) {
            break;
        }

        waiting_queue_.pop_front();
        block_manager_.allocate(front_id, req_length);

        Request& r = requests_.at(front_id);
        r.state = RequestState::Running;
        admitted_seqs.push_back(front_id);
        tokens_admitted += req_length;
    }

    if (!admitted_seqs.empty()) {
        running_list_.insert(running_list_.end(), admitted_seqs.begin(), admitted_seqs.end());
        Step step{true, admitted_seqs};
        pending_step_ = true;
        pending_step_obj_ = step;
        return step;
    }

    if (running_list_.empty()) {
        return Step{false, {}};
    }

    std::vector<SeqId> decode_seqs;
    size_t idx = 0;

    while (idx < running_list_.size()) {
        SeqId s = running_list_[idx];
        std::vector<SeqId> preempted_this_step_for_s;

        while (!block_manager_.can_append(s)) {
            if (running_list_.empty()) {
                break;
            }

            SeqId victim = running_list_.back();
            running_list_.pop_back();

            block_manager_.free(victim);
            Request& v_req = requests_.at(victim);
            v_req.preemptions++;
            v_req.state = RequestState::Waiting;
            preempted_this_step_for_s.push_back(victim);

            if (victim == s) {
                break;
            }
        }

        for (SeqId v : preempted_this_step_for_s) {
            waiting_queue_.push_front(v);
        }

        bool s_preempted = false;
        for (SeqId v : preempted_this_step_for_s) {
            if (v == s) {
                s_preempted = true;
                break;
            }
        }

        if (!s_preempted) {
            block_manager_.append_slot(s);
            decode_seqs.push_back(s);
            idx++;
        }
    }

    Step step{false, decode_seqs};
    pending_step_ = true;
    pending_step_obj_ = step;
    return step;
}

void Scheduler::update(std::span<const int32_t> tokens) {
    if (!pending_step_) {
        throw std::invalid_argument("[Scheduler] no pending step");
    }

    if (tokens.size() != pending_step_obj_.seqs.size()) {
        throw std::invalid_argument(
            "[Scheduler] tokens size does not match pending step sequences size");
    }

    for (size_t i = 0; i < pending_step_obj_.seqs.size(); ++i) {
        SeqId s_id = pending_step_obj_.seqs[i];
        int32_t token_val = tokens[i];

        auto it = requests_.find(s_id);
        if (it == requests_.end()) {
            throw std::invalid_argument("[Scheduler] sequence id not found");
        }

        Request& req = it->second;
        req.output.push_back(token_val);

        if (req.output.size() == 1 && req.first_token == Clock::time_point{}) {
            req.first_token = Clock::now();
        }

        bool finished = (config_.eos_id != -1 && token_val == config_.eos_id) ||
                        (req.output.size() >= size_t(req.max_new_tokens));

        if (finished) {
            block_manager_.free(s_id);
            req.state = RequestState::Finished;
            req.finished = Clock::now();

            auto r_it = std::find(running_list_.begin(), running_list_.end(), s_id);
            if (r_it != running_list_.end()) {
                running_list_.erase(r_it);
            }

            finished_list_.push_back(req);
        }
    }

    pending_step_ = false;
    pending_step_obj_ = Step{};
}

std::vector<Request> Scheduler::take_finished() {
    std::vector<Request> res = std::move(finished_list_);
    finished_list_.clear();

    for (const auto& req : res) {
        requests_.erase(req.id);
    }

    return res;
}

const Request& Scheduler::request(SeqId id) const {
    auto it = requests_.find(id);

    if (it == requests_.end()) {
        throw std::invalid_argument("[Scheduler] request id not held");
    }

    return it->second;
}

const BlockManager& Scheduler::blocks() const {
    return block_manager_;
}

int Scheduler::num_waiting() const {
    return static_cast<int>(waiting_queue_.size());
}

int Scheduler::num_running() const {
    return static_cast<int>(running_list_.size());
}

/*
TODO(JOE): Scheduler::cancel

Purpose. The HTTP server (`engine serve`, src/server/serving_loop.cpp) has to take a request out of the scheduler before it finishes: when its client disconnects in the middle of a stream, and when a stop string ends the answer. docs/api.md promises that a client that disconnects cancels its request and frees its blocks. Today a request leaves the scheduler only by EOS or max_new_tokens, so until this exists the server catches the logic_error below, prints one warning, stops sending the request's tokens anywhere and lets it decode until it finishes, holding its KV blocks and a batch row the whole time. With the dashboard's playground and several browser tabs that is real waste: an abandoned 2,000-token answer keeps 125 blocks and a row of every decode step for its whole length.

Interface. void Scheduler::cancel(SeqId id), declared in include/engine/sched/scheduler.h. It is called on the engine thread between steps, like submit: after update (and the take_finished that follows it) and before the next schedule. It removes request id from the scheduler wherever it is. A waiting request (never admitted, or preempted and back in the waiting queue) leaves the waiting queue; it holds no blocks. A running request leaves the running list and every block it holds goes back to the block manager at once. Afterwards request(id) throws, num_waiting() or num_running() is one lower, the id is free to be submitted again, and the request is never returned by take_finished.

What must hold. The other requests do not notice: the waiting queue and the running list keep their relative order, so the next schedule() admits, decodes and preempts exactly as if the cancelled request had never been submitted, and their outputs are unchanged. Blocks are conserved: once every request has finished or been cancelled, blocks().num_free() is num_blocks - 1 again (block 0 stays reserved). A request that was blocking the head of the waiting queue no longer blocks it, so a smaller request behind it can be admitted by the very next schedule().

Errors. Throw std::invalid_argument, naming the id, and change nothing, when id is not held (never submitted, already cancelled, or already returned by take_finished), when it has finished but has not been taken yet (state Finished), and when a step is pending (schedule() returned a non-empty step that update() has not consumed yet; cancelling then would make update's tokens line up with the wrong sequences). invalid_argument is a logic_error, so the server rethrows it instead of treating it as "not implemented".

Edge cases. Cancelling the only running request leaves an idle scheduler whose next schedule() returns an empty step. Cancelling right after submit, before any schedule(), is valid. Cancelling a preempted request while it waits must not free blocks a second time (preemption already freed them). Cancelling two requests in a row, or the one at the back of the running list (the next preemption victim), keeps the order of the rest.

Tests that prove it. tests/todo/unit/test_scheduler_cancel.cpp (target todo_unit_tests, label todo_joe): a waiting request leaves the queue, a running one frees all its blocks at once, only the other blocks stay held, the other requests keep their batch order, a cancelled request is never returned as finished, freeing the front of the queue admits the next request, a preempted request can be cancelled while it waits, unknown, cancelled, finished-but-not-taken and taken ids are rejected, a cancel while a step is pending is rejected and changes nothing, the id can be submitted again, and 20 random runs of 12 requests with random cancels keep every other answer equal to its answer alone and return every block. Run them with ctest --preset cpu -L todo_joe. When they pass, move the file into tests/unit/, add it to unit_tests and delete todo_unit_tests; tests/unit/test_serving_loop.cpp and tests/unit/test_http_server.cpp then exercise the server path where blocks come back right after a disconnect, and the server's warning disappears.
*/
void Scheduler::cancel(SeqId) {
    throw std::logic_error("unimplemented: Scheduler::cancel");
}

}  // namespace engine
