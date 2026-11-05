# Branch 83: sched/chunked-prefill-wiring (issue #101, Part B)

Worktree `~/code/llm-inference-engine-chunked`, branch `sched/chunked-prefill-wiring` from origin/main cbffd5b. Five local commits, unpushed. Each one was gated alone: the tree held exactly that commit's content when its gate ran. Every commit message is one line, a blank line, then `Co-Authored-By: Claude <noreply@anthropic.com>`, as the parent session asked.

## Commits

- 783e6aa Build paged batches for chunked prefill steps (Part B)
- b390aac Run paged batches that sample no sequence (Part B)
- 0614551 Serve chunked prefill steps under a token budget (Part B)
- 597f0a8 Add --token-budget to engine bench (Part A, CLI plumbing)
- 2a63485 Add --token-budget to engine serve (Part A, CLI plumbing)

## Headline finding: outputs match up to BF16 ties, not token for token

Greedy outputs do not depend on the budget except at BF16 near-ties. Across budgets they are not identical token for token, and they cannot be with cuBLAS: a chunk changes the GEMM row count, cuBLAS then picks a different kernel and summation order, and that moves logits by an ulp or so. This is the same effect technical_notes.md records for prefill against token-by-token decode. Measured on the 16 reference prompts (22 to 75 tokens, 48 new tokens, max batch 16), counting requests identical to whole-prompt prefill: budget 16 (= max_batch): 16 of 16; budget 23: 13; budget 40: 15; budget 100: 14; budget 2,048: 16. I checked every divergence against the contiguous ModelRunner's logits. Each one is a tie, with both tokens within 1 to 2 BF16 ulps of the best logit. Examples: request 6 at budgets 40 and 100 diverges at token 0, logits 15.6875 against 15.75, ulp 0.0625. Request 9 at budget 23 diverges at token 5, 16.5 against 16.625, ulp 0.125. The tests therefore use the repo's existing rule from `expect_equal_up_to_a_tie`: identical, or the first divergence is a tie within 2 ulps. They print the identical counts. More evidence that the logic is right: the new runner test compares every sampled row of chunked steps with the contiguous runner and gets a worst relative L2 of 0.021, against 0.028 for the existing unchunked mixed-batch test. Five deliberate mutants of the builder and runner each fail the new tests (logit row on the first row of a chunk, chunk rows before decode rows, tokens shifted by one, chunk positions restarting at 0, skipping the layers when nothing is sampled).

## What changed

- `build_chunked_batch(ChunkedStep, Scheduler, max_blocks)` in src/sched/paged_batch.cpp puts decode rows first (one each, last token), then the rows of each chunk (positions first to first + count - 1). Logit rows go only on rows that reach their sequence's last token, and the builder checks that those sequences equal `sampled`, in order. `build_paged_batch` now shares the per-sequence row code (`append_rows`). `sample_rows` takes a `span<const SeqId>`, so the same function serves `Step::seqs` and `ChunkedStep::sampled`.
- PagedRunner::stage and run_staged accept `num_seqs == 0`. The layers still run, because they write the chunk's KV. The final norm and lm_head are skipped, and the result is a `{0, vocab}` tensor. Sampler returns at once for zero rows (the sample kernel would otherwise launch an empty grid).
- `serve_step(..., trace, token_budget = 0)` and `serve(..., graphs, token_budget = 0)`: 0 keeps `schedule()`; above 0 the step comes from `schedule_chunked`. `update` is always called, with an empty token list when nothing was sampled. Pure decode steps still use DecodeGraphs, because `batch.prefill` is false and the logit rows are the identity. Mixed and chunk-only steps run eager.
- `run_batched(..., token_budget)`, `engine bench --token-budget N`, `GpuBackendOptions::token_budget` and `engine serve --token-budget N`. The value must be at least `--max-batch`, and bench rejects it without `--max-batch`. The paged runner is sized to `max(max_prefill, budget)` tokens. The bench summary CSV, the trace meta and the serve HTTP meta record `token_budget`. README and docs/api.md describe the flag and the trace semantics.
- `generate` never drives the scheduler, so nothing changed there. `scripts/checksum.py` and the checksum golden tests are untouched and green in the gate.

## Verification

- Gate script `~/.cache/llmie-chunked/verify.sh` (copy of the server one, cd into this worktree). GPU was up for every run. Final lines for each commit: 783e6aa debug 698/698, cpu 313/313; b390aac debug 699/699, cpu 313/313; 0614551, 597f0a8 and 2a63485 debug 701/701, cpu 313/313; all VERIFY OK. The first gate attempt for b390aac failed once in `FastSwiglu.RowResultsDoNotDependOnTheBatch` with `cudaErrorInvalidValue` from `cudaStreamCreate` in tests/support/device.h. That is unrelated code and the usual GPU flakiness: it passed 5 of 5 on rerun, and the full gate rerun was green before the commit.
- CLI, debug build: `engine bench --max-batch 4 --prompts scripts/prompts.json --num-requests 6 --output-len 12` at budgets 0, 16 and 40 gave identical traced outputs. Budget 16 had two steps that sampled nothing (empty `seqs` and `tokens`). `--token-budget 3` with `--max-batch 4` and `--token-budget 16` without `--max-batch` both exit with a clear error. `engine serve --max-batch 4 --graphs` with budgets 0 and 16 gave identical replies to three concurrent greedy chat requests (38-token prompt, 40 tokens out).
- Logs: `~/.cache/llmie-chunked/` (gate.log, ctest_*.log, gpu_serve.log, gpu_tie.log, bench_*.csv, trace_*.json, chat_*.json).

## Not verified

- compute-sanitizer memcheck on the new GPU tests.
- Release build and any performance number: no TPOT or TTFT measurement and no QPS sweep (deferred to the follow-up branch as instructed).
- Sampled (temperature) outputs under a budget; only greedy is tested across budgets.
- The web dashboard in live mode against `serve --token-budget` (steps with empty `seqs`, and `kind` for mixed steps).
- The `AStepTraceRebuildsEveryOutputThroughPreemption` test was not repeated in chunked mode. The bench trace check above covers the rebuild without preemption.

## Open questions for Joe

1. Budget invariance criterion. You asked for token-for-token equality, which is not reachable for budgets that change the GEMM row counts (evidence above). I used the existing 2-ulp tie rule. Budgets 16 and 2,048 happen to be exact today, but asserting that would rely on luck, so I did not.
2. Meaning of `prefill` in traces and stats for mixed steps. I set it to "the step has no decode rows" (`Planned::prefill = step.decode.empty()`, src/runtime/serve.cpp:37), so `ServingLoop`'s preemption counter (`trace.prefill && live.generated > 0`) never counts decode rows as preemptions. The cost: a preempted sequence that finishes its replay inside a mixed step is not counted in `preemptions_total`, and `prefill_steps` only counts steps with no decode rows. Recommendation: count preemptions from `Request::preemptions` in a later branch.
3. Commit trailer. `subagent_rules.md` says no trailer; the parent session passed your override to add `Co-Authored-By: Claude`, and I followed the override.
4. Part tags: the builder, runner and serve_step commits are Part B; the two CLI commits are Part A, following working_rules.md ("CLI and benchmark harness plumbing").

## Subtle lines for review

- src/sched/paged_batch.cpp:18 `append_rows` returns whether it added a logit row (line 38, `first + count == length`). build_paged_batch ignores the return value, because prefill and decode rows always reach the end.
- src/sched/paged_batch.cpp:67 `batch.prefill = !step.prefill.empty()`. This keeps any step with chunk rows off the graph path, even a mixed step whose shape would pass `covers`.
- src/sched/paged_batch.cpp:77 `ends == step.sampled` is the single check that logit rows and sampled entries line up in order.
- src/runtime/paged_runner.cpp:122 `identity = logit_rows.empty() && s > 0`, so `s == 0` with no logit rows passes the count checks. Line 182 returns after all layers have run.
- src/runtime/sampler.cpp:21 early return for zero rows.
- src/runtime/serve.cpp:37 against paged_batch.cpp:67: `Planned::prefill` (trace and stats) and `PagedBatch::prefill` (graph eligibility, NVTX label) mean different things for mixed steps on purpose.
- src/server/gpu_backend.cpp:64 and src/app/bench_command.cpp:187: the runner holds `max(max_prefill, budget)` tokens, so a budget above 2,048 grows the activation buffers.

## Commands for Joe

```bash
cd ~/code/llm-inference-engine-chunked && bash ~/.cache/llmie-chunked/verify.sh
```

```bash
./build/debug/bin/gpu_tests --gtest_filter='ServeTest.TokenBudget*:PagedRunnerTest.ChunkedSteps*'
```

```bash
./build/debug/bin/unit_tests --gtest_filter='ChunkedBatch.*:PagedBatch.*'
```

```bash
compute-sanitizer --tool memcheck --leak-check full ./build/debug/bin/gpu_tests --gtest_filter='PagedRunnerTest.ChunkedSteps*'
```

## Draft PR description

Title: Wire chunked prefill into serving (--token-budget)

Wires `Scheduler::schedule_chunked` into serving for #101. `engine bench` and `engine serve` take `--token-budget N` (at least `--max-batch`). Each step then holds at most N tokens: every decoding sequence gets its row first, and the rest goes to prompts, which split across steps. Without the flag nothing changes.

- `build_chunked_batch`: decode rows, then chunk rows; logit rows only for the sampled sequences, in order.
- `PagedRunner` runs batches that sample nothing (a step of unfinished chunks writes KV and returns no logits); `Sampler` skips zero rows.
- `serve_step`, `serve` and `run_batched` take the budget. `update` runs even with no tokens. Pure decode steps keep their CUDA Graph, and mixed steps run eager.
- Bench summary, trace meta and serve meta record `token_budget`. README and docs/api.md are updated.

Correctness: on the 16 reference prompts, greedy outputs at budgets 16, 23, 40, 100 and 2,048 equal whole-prompt prefill (16, 13, 15, 14 and 16 of 16 identical) or first differ at a tie within 2 BF16 ulps of the best logit. Chunk sizes change GEMM row counts and with them cuBLAS summation order. The same holds through forced preemption, with and without graphs. Chunked steps also match the contiguous runner row by row (worst relative L2 0.021).

Not in this PR: the QPS sweep before and after, which comes in the follow-up branch.
