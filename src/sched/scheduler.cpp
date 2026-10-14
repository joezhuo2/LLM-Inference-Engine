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

}  // namespace engine
