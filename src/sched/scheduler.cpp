#include "engine/sched/scheduler.h"

#include <stdexcept>

namespace engine {

/*
TODO(JOE): Scheduler

Purpose. The scheduler is the continuous (iteration-level) batching policy of the engine: it holds the waiting, running and finished requests, decides before every forward pass which sequences take part and whether that pass is a prefill or a decode, keeps the block manager in step with the tokens the runner is about to write, and takes the sampled tokens back afterwards. Requests join the batch as soon as there is room and leave as soon as they finish, instead of the whole batch waiting for its slowest member, which is where batching's throughput comes from. When the KV cache runs out it preempts by recompute: the victim's blocks are freed and it is later prefilled again over its prompt plus everything it has generated, which gives the same tokens because the model is deterministic and the sampling step is the number of tokens generated so far. It is pure C++ with no CUDA, so the whole policy is tested on the CPU with a fake model; the model runner (branch 20) is the only thing that turns its steps into forward passes.

Ownership. Scheduler(config) creates and owns its BlockManager(config.num_blocks, config.block_size); blocks() gives read access to it, and nothing else may change it. The runner reads block tables, slots and lengths from blocks() to build block_tables, slot_mapping and context_lens. Requests are owned by the scheduler from submit until take_finished; request(id) gives read access to a waiting, running or finished (not yet taken) request.

Configuration. num_blocks and block_size are passed to the block manager (its own checks apply, with its std::invalid_argument). max_batch is the most sequences in one step, max_prefill_tokens the most tokens one prefill step may feed (the build plan's 2,048 budget), max_context the longest sequence a request may reach (what validate(request, max_context) checks), and eos_id the token that ends a request (the default -1 means none, since token ids are never negative). The constructor requires max_batch >= 1, max_context >= 1, max_prefill_tokens >= max_context, and a cache that holds one sequence of max_context tokens on its own, num_blocks - 1 >= ceil(max_context / block_size). Together with validate these guarantee that a request at the front of the queue always fits once nothing else is running, so the scheduler can never stall.

Lifecycle. submit(request) checks validate(request, config.max_context) and that request.id is not already held by the scheduler (waiting, running or finished and not yet taken), then sets state = Waiting, preemptions = 0 and arrived = Clock::now(), and appends the request to the back of the waiting queue; any other value the caller left in those three fields is overwritten. It may be called at any time, including between schedule() and update(). A request moves to Running when it is admitted, back to Waiting when it is preempted, and to Finished when update() gives it its last token. take_finished() returns the finished requests in the order they finished, by value, and forgets them, after which request(id) throws and the id may be submitted again.

One step: schedule(). Every call returns a Step whose seqs are the sequences of the next forward pass, in the order of the logits rows the runner will produce, and either prefill is true and every sequence feeds all of its tokens, or prefill is false and every sequence feeds only its newest token (the MVP never mixes the two). First, admission: while the waiting queue is not empty, fewer than max_batch sequences are running, the front request's length L = prompt.size() + output.size() fits the free blocks (blocks().can_allocate(L)), and L plus the tokens already admitted in this call is at most max_prefill_tokens, the front request is allocated L tokens in the block manager, becomes Running, and is appended to the step. Admission is strictly first come, first served: it stops at the first request that does not fit and never skips ahead to a smaller one behind it. If at least one request was admitted, the step is a prefill of exactly the admitted requests, in admission order, and nothing else changes in this call. Otherwise, if any sequence is running, the step is a decode of every running sequence, built as described under Preemption. Otherwise the scheduler is idle and the step is empty (prefill false, no seqs). After a non-empty step, for every sequence s in seqs, blocks().length(s) equals request(s).length(), so the tokens to feed are positions 0 to length - 1 for a prefill and position length - 1 for a decode, and blocks().slot(s, pos) is where each one's K and V go.

Preemption. A decode step visits the running sequences from the earliest admitted to the most recently admitted (a re-admitted request counts as admitted when it was re-admitted). Before sequence s takes its slot, as long as blocks().can_append(s) is false, the most recently admitted sequence still running is preempted, which may be s itself, in which case s is not part of this step. Preempting a sequence frees all of its blocks, keeps its output, increments its preemptions count, sets its state to Waiting, and puts it at the front of the waiting queue; several sequences preempted in one call end up at the front in the order they were admitted, ahead of every request that was already waiting. Once s can append, it takes one slot (blocks().append_slot(s)) and joins the step. A preempted request is admitted again later like any other waiting request, with L counting its prompt and its output so far, so its prefill recomputes the K and V of every token it had, and its next sampled token continues where it stopped. The configuration checks above guarantee that the earliest admitted sequence never needs to preempt itself, so a decode step is never empty while anything is running.

Taking tokens back: update(tokens). After every non-empty step, the caller passes exactly one sampled token per sequence of that step, in the same order, and update appends each token to its request's output in that order. When a request's output reaches length 1, first_token is set to Clock::now(); a preempted request keeps its first_token. A request whose new token equals eos_id, or whose output now holds max_new_tokens tokens, is finished: its blocks are freed, its state becomes Finished, finished is set to Clock::now(), it leaves the running set, and it joins the finished list (several in one call keep the step's order). The token that ended a request, EOS included, stays in its output, as in generate(). Token values are not checked. Every other sequence of the step stays running with blocks().length(s) equal to request(s).length() - 1, because its newest token has no K and V yet; the next step it takes part in feeds it.

Errors. Every violated precondition throws std::invalid_argument with a message that names the offending request id or argument, and leaves the scheduler exactly as it was (no request added, admitted, preempted or changed, no blocks taken or freed). The preconditions are: the constructor's checks above; validate and a fresh id for submit; for schedule, that the previous non-empty step has been given its update; for update, that a non-empty step is pending and tokens.size() equals its number of sequences; a held id for request. An empty step needs no update. std::invalid_argument is required, not just any std::logic_error, so a rejection test cannot pass against this stub. With these preconditions met, no BlockManager call made by the scheduler may ever throw.

Invariants. Between calls: every request is in exactly one of the waiting queue, the running set and the finished list, and its state says which; at most max_batch sequences run; exactly the running sequences are known to the block manager, so its free blocks plus the blocks of all running tables equal num_blocks - 1 (waiting, preempted and finished requests hold no blocks); a running sequence that is not part of a pending step holds request.length() - 1 tokens in the block manager; arrived <= first_token <= finished once they are set. The scheduler is deterministic: two schedulers given the same calls with the same tokens return the same steps.

Edge cases and boundaries. A request with max_new_tokens = 1 finishes in the update that follows its prefill and never decodes. A request whose first token is EOS finishes the same way. A prompt of exactly max_context tokens is admitted with max_new_tokens = 1. A prefill whose total is exactly max_prefill_tokens is allowed, one more token is not. A cache that is exactly full after admission is fine; the next decode step preempts as needed. Requests submitted while a step is pending wait for the next schedule().

What to compare against. A fake model in the test: a fake KV cache of token ids that the test fills through blocks().slot() exactly as the runner will write K and V, and a next-token function of the whole context read back through blocks().table(), the seed and the step. Each request's output must equal what the same fake model produces for that request on its own, which is the build plan's "16 concurrent requests produce the same text as 16 sequential runs", and it must stay equal when a tiny cache forces preemption. The real-model version of that check arrives with the paged model runner in branch 20.

Tests that prove it. tests/todo/unit/test_scheduler.cpp: the constructor and submit checks, an idle scheduler, first come first served admission up to max_batch and the prefill budget, admission stopping at the first request that does not fit the free blocks, prefill before decode, the block manager lengths after each step, retiring on EOS and on max_new_tokens with the blocks freed, max_new_tokens = 1, an exact preemption scenario (victim choice, a sequence preempting itself, the waiting queue order, the recomputed prefill), the timestamps, take_finished and request(id), every rejected call leaving the state unchanged, determinism, 16 concurrent requests against 16 solo runs with a large cache and with a tiny one that forces preemption, and 10,000 random steps with random submissions checked against the invariants after every call. They run in the todo_unit_tests target with the todo_joe label (ctest --preset debug -L todo_joe) and also need your BlockManager. When they all pass, move the file to tests/unit/, add it to unit_tests and delete todo_unit_tests if it is empty.
*/
Scheduler::Scheduler(const SchedulerConfig&) {
    throw std::logic_error("unimplemented: Scheduler");
}

void Scheduler::submit(Request) {
    throw std::logic_error("unimplemented: Scheduler");
}

Step Scheduler::schedule() {
    throw std::logic_error("unimplemented: Scheduler");
}

void Scheduler::update(std::span<const int32_t>) {
    throw std::logic_error("unimplemented: Scheduler");
}

std::vector<Request> Scheduler::take_finished() {
    throw std::logic_error("unimplemented: Scheduler");
}

const Request& Scheduler::request(SeqId) const {
    throw std::logic_error("unimplemented: Scheduler");
}

const BlockManager& Scheduler::blocks() const {
    throw std::logic_error("unimplemented: Scheduler");
}

int Scheduler::num_waiting() const {
    throw std::logic_error("unimplemented: Scheduler");
}

int Scheduler::num_running() const {
    throw std::logic_error("unimplemented: Scheduler");
}

}  // namespace engine
