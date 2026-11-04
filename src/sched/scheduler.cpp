#include "engine/sched/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
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

/*
TODO(JOE): Scheduler::schedule_chunked

Purpose. schedule() makes every step all prefill or all decode, and prefill goes first, so one long prompt stalls every running sequence: the batch-16 profile's 113 ms prefill costs about 12 decode steps, and a 2,048-token prompt takes about 1.6 s alone (docs/results.md). Under Poisson load vLLM reaches 6.1 req/s of goodput against the engine's 3.2 with the same token ids and arrivals, and prefill stalls are the likely cause, not yet measured on their own (#101). This method gives each step a token budget, gives every decoding sequence its row first, and spends what is left on prefill, so a long prompt spreads over several steps while the others keep decoding. schedule() stays as it is; a scheduler uses one of the two for its whole life, and nothing calls this one until the runner wiring lands.

Interface. ChunkedStep Scheduler::schedule_chunked(int token_budget), declared in include/engine/sched/scheduler.h with PrefillChunk and ChunkedStep. decode lists the sequences that get one row each, for their last token, in running order. prefill lists the chunks in the order their rows go into the batch, after the decode rows: chunk {id, first, count} covers positions first to first + count - 1 of request id, in the numbering of Request::token. sampled lists the sequences that need a token from this step, in row order: every decode entry, then each sequence whose chunk reaches the end of its length. A chunk that does not reach the end has no entry and no logits. The existing update(tokens) consumes the step and takes exactly one token per sampled entry, in that order. It may therefore be empty, and an empty update still has to be called: a step with only unfinished chunks is pending like any other non-empty step.

Filling a step. First, every running sequence whose prefill is complete gets a decode row, exactly as schedule() does it: append a slot, and when can_append is false preempt from the back of the running list, with the same rule for the victim, the waiting queue and Request::preemptions. Then the rest of the budget goes to the sequence still in prefill, of which there is at most one, because the next is admitted only when budget is left over. Then, while budget remains, waiting requests are admitted first come, first served, up to max_batch, stopping at the first one can_allocate refuses and never skipping it. num_tokens() never exceeds token_budget, and whenever a sequence is left with unfinished prefill the step has used the whole budget.

Blocks and progress. Admission allocates blocks for the request's whole length at once, as schedule() does, so a sequence in prefill can never run out of blocks halfway and blocks().length(id) equals request(id).length() from its first chunk on. The sequence state is Running from admission. Chunks of one sequence are contiguous and start at 0. Where the progress is kept is up to you, but it has to be forgotten in every way a sequence can leave: preemption, finishing, and cancel. A preempted sequence, whether it was decoding or still in prefill, starts again at position 0 over prompt plus output, like schedule()'s prefill, and an id submitted again after a cancel starts at 0 too.

What must hold. Outputs do not depend on the budget: with a correct KV cache every answer equals the request's answer alone, since Request::output and the KV contents are all a sequence's sampling depends on. A sequence whose prefill is complete and that was not preempted gets a row in every step, which is the point of this work. The sequence in prefill is always the last in the running list, so preemption takes partly prefilled work first. first_token is stamped by update when the first token arrives, not at the first chunk, and a request with max_new_tokens = 1 finishes on the step of its last chunk and frees its blocks then. max_prefill_tokens does not apply here: the budget is the limit, and a prompt longer than the budget is the case this exists for.

Errors. Throw std::invalid_argument and change nothing when token_budget is below config_.max_batch (below that, a full batch of decodes could leave a sequence in prefill with no tokens forever), and when a step is pending, as schedule() does. cancel while a chunk-only step is pending must be rejected as well.

Edge cases. An idle scheduler returns an empty step that is not pending. A prompt exactly the size of the remaining budget finishes in that step and decodes in the next. A budget equal to max_batch still advances a sequence in prefill by at least one token per step, because it holds one of the max_batch rows. A sequence with one prompt token left is still in prefill: its last row is a chunk, needs no new slot, and is not a decode row. A chunk-only step changes no output and no length. Several requests can share a step: one finishing its last chunk and the next starting its first.

What is not here. The batch builder for a mixed step, serve_step, the sampler and a CLI flag come in the next branch, once this passes. PagedRunner::stage and run_staged reject a batch with zero sequences today, which is what a chunk-only step produces, and DecodeGraphs::covers only takes pure decode batches, so mixed steps run eager and lose about 1 ms each unless that changes. Issue #101 asks for p99 TPOT under --request-rate before and after.

Tests that prove it. tests/todo/unit/test_scheduler_chunked.cpp (target todo_unit_tests, label todo_joe). Run them with ctest --preset cpu -L todo_joe. When they pass, move the file into tests/unit/, add it to unit_tests and delete todo_unit_tests.
*/
ChunkedStep Scheduler::schedule_chunked(int) {
    throw std::logic_error("unimplemented: Scheduler::schedule_chunked");
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

void Scheduler::cancel(SeqId id) {
    const std::string name = std::to_string(id);

    if (pending_step_) {
        throw std::invalid_argument("[Scheduler] cannot cancel request " + name +
                                    " while a step is pending");
    }

    auto it = requests_.find(id);
    if (it == requests_.end()) {
        throw std::invalid_argument("[Scheduler] request " + name + " not held");
    }

    if (it->second.state == RequestState::Finished) {
        throw std::invalid_argument("[Scheduler] request " + name + " already finished");
    }

    if (it->second.state == RequestState::Waiting) {
        waiting_queue_.erase(std::find(waiting_queue_.begin(), waiting_queue_.end(), id));
    } else {
        block_manager_.free(id);
        running_list_.erase(std::find(running_list_.begin(), running_list_.end(), id));
    }

    requests_.erase(it);
}

}  // namespace engine
