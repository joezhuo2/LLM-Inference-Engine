#include "engine/sched/speculative.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/kv/block_manager.h"

namespace engine {

/*
TODO(JOE): speculative decoding by prompt lookup (propose_prompt_lookup, count_accepted, BlockManager::truncate)

Purpose. Issue #102. A decode step produces one token per sequence, and at batch 1 most of its time is reading the weights once for a single row. Speculative decoding guesses the next k tokens cheaply, runs the model once on all of them as k + 1 rows of the same sequence (the weights are still read once), and keeps the guesses the model agrees with. The plan (docs/claude/build_plan.md, P13) starts with prompt lookup: no draft model, the guess is copied from the place in the context where the last few tokens appeared before. It pays off when the output repeats its input (summaries, code edits, quoting) and costs k wasted rows per step when it does not. The output is exactly what plain decoding would produce, token for token, because every kept token is the one the model sampled. These three functions are the host side: making the guess, counting how much of it survived, and giving back the cache slots of the part that did not.

The step they serve. A sequence has its tokens 0 to n - 1, with keys and values cached for 0 to n - 2, and x = token n - 1 is the input of the next step. Plain decode runs one row (x at position n - 1) and samples token n. A speculative step takes draft = propose_prompt_lookup(tokens 0 to n - 1), of length k, reserves k + 1 slots with append_slot, and runs k + 1 rows: x at position n - 1, then draft[i] at position n + i. Row i attends to everything before it, like a prefill chunk, and is sampled with the step number of output token i of this step (the sampler is keyed by seed and step, include/engine/sampling/sample.h), giving sampled[0 .. k]. sampled[0] is always right: it is what plain decode would have sampled. sampled[i + 1] is right only if draft[0 .. i] were all right, because row i + 1 was computed from them. With a = count_accepted(draft, sampled), the step emits sampled[0 .. a], that is a + 1 tokens, the cache is valid for positions up to n - 1 + a, and truncate(seq, n + a) gives back the slots of the k - a rows that were computed from a wrong token. sampled[a] is the input of the next step.

Interface. Declared in include/engine/sched/speculative.h and include/engine/kv/block_manager.h. propose_prompt_lookup(context, max_ngram, max_draft) returns at most max_draft tokens. For g from min(max_ngram, context.size() - 1) down to 1, take the last g tokens of context as the pattern and look for an earlier start i where context[i .. i + g) equals the pattern and i + g <= context.size() - 1 (the match is not the pattern itself and has at least one token after it; it may overlap the pattern). The first g that has a match wins, and among its matches the latest one (largest i) wins. The result is context[i + g .. min(i + g + max_draft, context.size())). No match for any g gives an empty vector, as does a context of fewer than 2 tokens. count_accepted(draft, sampled) returns the largest a such that draft[i] == sampled[i] for all i < a; sampled may be shorter or longer than draft (normally draft.size() + 1), and a never exceeds either size. BlockManager::truncate(seq, length) sets the length of seq to length and releases the blocks its table no longer needs, so the table has ceil(length / block_size) entries afterwards; the next append_slot returns the slot of position length, reusing the slot that was given back.

Semantics of truncate. A released block follows the rules of free for one block: its count goes down by 1, and at 0 it becomes blank (or cached, if it is published, once prefix caching from #99 is in). truncate to the current length does nothing. The caller guarantees that no position at or after length has been published: the rows of a speculative step are published, if at all, only after truncate. That is what keeps a partial last block private.

Errors. Throw std::invalid_argument when: max_ngram < 1 or max_draft < 1 in propose_prompt_lookup (message naming the function); seq is not registered, length < 1 or length > length(seq) in truncate (message starting with [BlockManager]; to drop a whole sequence use free). count_accepted cannot fail.

Edge cases. A context where the pattern only occurs as itself gives nothing. A context of one repeated token gives a draft of 1 token (the latest match ends one token before the end). max_ngram above the context length is allowed. Token ids are compared as plain numbers; there is no special token handling here: stopping at EOS or at max_new_tokens inside the accepted run is the scheduler's job.

Cost. Looking for the pattern by scanning the context is O(context.size() * max_ngram) per sequence per step, a few thousand comparisons at this engine's context sizes. Measure before building an index.

What to compare against. The tests check propose_prompt_lookup against a plain triple loop written out in the test file over random contexts with a small vocabulary. For the feature as a whole, once it is wired: the same prompt and seed give the same tokens with and without speculation, the acceptance rate (accepted drafts / proposed drafts) and tokens per step, and TPOT on prompts that repeat their input (a summary of a long text, a code edit) and on chat prompts that do not, each against plain decode. Put both in docs/results.md.

What is not here. The scheduler and runner side, a separate branch once these pass: a step type whose sequences carry draft rows (Scheduler::schedule_chunked already has prefill chunks with several rows per sequence; a speculative sequence is a chunk in which every row is sampled), reserving k + 1 slots and falling back to plain decode when blocks are short or the draft is empty, sampling every row with its own step number, calling truncate in update, stopping at EOS inside the accepted run, counting proposed and accepted tokens in the metrics, and a flag on engine serve and engine bench. Decode graphs are keyed by row count (#121), so a speculative step needs its own buckets or runs outside the graph. A draft model instead of prompt lookup is a later issue.

Tests that prove it. tests/todo/unit/test_speculative.cpp (target todo_unit_tests, label todo_joe), 24 tests checked against a throwaway implementation and eight mutants of it: the longest pattern, the latest match, overlap, the end of the context, 4,000 random contexts against a plain search, the accept count, and truncate at every accept count of a 9-row step. Run them with ctest --preset cpu -L todo_joe. When they pass, move the file to tests/unit/, add it to unit_tests and take it out of todo_unit_tests.
*/
std::vector<int32_t> propose_prompt_lookup(std::span<const int32_t>, int, int) {
    throw std::logic_error("unimplemented: propose_prompt_lookup");
}

int count_accepted(std::span<const int32_t>, std::span<const int32_t>) {
    throw std::logic_error("unimplemented: count_accepted");
}

void BlockManager::truncate(SeqId, int) {
    throw std::logic_error("unimplemented: BlockManager::truncate");
}

}  // namespace engine
