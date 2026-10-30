# M3 plan: paged KV and batching (due Nov 1)

Exit criteria (from `build_plan.md`): block manager with unit tests; paged decode attention kernel matches the naive kernel; the scheduler serves 16 concurrent requests with correct outputs for each (16 concurrent requests produce the same text as 16 sequential runs, also under forced preemption).

## Ownership in M3

Most of M3's core is Part C (Joe writes it; Claude stubs, specs and tests): the KV block manager, the KV cache layout and `write_kv` slot mapping, paged decode and prefill attention, the scheduler and preemption. Part B (Claude drafts): batched sampling (per-row seed, step and temperature), top-k and top-p, and wiring the paged cache and batches into `ModelRunner`. Part A (Claude): the request abstraction, KV cache sizing from `cudaMemGetInfo`, the bench harness and CSV output.

## Branches (order confirmed by Joe on Oct 10)

| # | Branch | Part | Content | State |
| --- | --- | --- | --- | --- |
| 15 | kv/block-manager | C stub | `BlockManager` stub with TODO(JOE) spec, 15 todo_joe tests, `todo_unit_tests` target | Merged (#24) |
| 16 | kv/paged-cache | C stub and A | KV cache layout and `write_kv` stub with spec and todo_gpu tests (new `todo_gpu_tests` target); cache sizing from free memory (Part A) | Merged (#25, squashed by Joe into one commit) |
| 17 | kernel/paged-decode-attention | C stub | Paged decode attention stub, tests against `naive::attention` on shuffled block tables, context lengths 1, 15, 16, 17, 2,048 | Merged (#26) |
| 18 | sampling/batched | B | Per-row seed, step and temperature in `sample` (the batch row dropped from the noise), top-k and top-p by threshold search, `--top-k`/`--top-p` | Merged (#27) |
| 19 | sched/scheduler | C stub and A | `Request` (Part A), scheduler and preemption stub with spec, the 16-request and forced-preemption tests (CPU, fake model) | Done locally (3 commits), not pushed |
| 20 | model/paged-runner | B | `PagedBatch` builder, `PagedRunner` on the paged cache, `serve()` loop, M3 criterion on the real model | Done locally (3 commits), not pushed |
| 21 | tools/bench | A | `engine bench`, synthetic workloads, per-request CSV | Done locally (4 commits, batch 1 driver), decisions confirmed, not pushed |

Paged prefill attention (Part C) can come after 17; until then prefill may keep a contiguous path or use the decode kernel per query row, to be decided with Joe when branch 20 starts.

## Decisions in branch 15 (Claude's, open to Joe's change)

- `SeqId` is `int64_t` (the request id); the block manager does not interpret it.
- Block 0 is the null block, so usable capacity is `num_blocks - 1` and the build plan's invariant `num_free + owned == num_blocks` becomes `num_blocks - 1`.
- Every violated precondition throws `std::invalid_argument` with the strong guarantee; `invalid_argument` (not plain `logic_error`) so rejection tests cannot pass against the stub.
- Added `length(seq)` beyond the build plan's sketch (the runner needs `context_lens`). Left out `refs_` (copy-on-write is Part 2) and the private members (Joe's choice).
- Allocation order is unspecified but must be deterministic.
- Enforcing the 2,048 context limit is the scheduler's job, not the block manager's.

## Decisions in branch 16 (Claude's, open to Joe's change)

- `write_kv(k_cache, v_cache, qkv, slot_mapping, stream)` in namespace `engine` (not `naive`); K and V are found from the cache shape alone, like `store_kv`, so no head counts are passed.
- Slot `-1` is padding; slot values are not validated (would need a device-to-host copy); duplicate or out-of-range slots are undefined.
- The kernel stub aborts (per the delegation), so todo_joe tests show as "Subprocess aborted"; each ctest test runs in its own process, so one abort does not hide the others.
- `KvCache` is one allocation laid out layer 0 K, layer 0 V, layer 1 K, ...; not zeroed (debug builds poison it with `0xFF`, BF16 NaN). `kv_cache_blocks` takes `fraction = 0.9` of the free bytes passed in and caps at `INT_MAX`; fewer than 2 blocks throws `runtime_error`.

## Decisions in branch 17 (Claude's, open to Joe's change)

- `KvCache` stays Part A (Joe chose option 1 on Oct 10): the shape is a build plan decision, and the indexing lives in Joe's `write_kv` and attention.
- `paged_attention_decode(out, qkv, k_cache, v_cache, block_tables, context_lens, stream)` in namespace `engine`; heads from `out`, everything else from the cache shape; only Q columns of `qkv` are read.
- Scores rounded as HF does (`bf16(bf16(q . k) * scale)`), probabilities free; tolerance 1/64 of `sum p|v| + |naive|` against `naive::attention`, and HF's `attn` within `2^-7 * max|v|`.
- Context length 0 is padding (row not written); output bit-identical regardless of memory outside the context and of batch composition; table capacity above 2,048 may be rejected.

## Decisions in branch 18 (Claude's, open to Joe's change)

- Per-row parameters travel as a device array of `SampleRow { SamplingParams params; uint64_t step; }`; `Sampler` (runtime) validates on the host and uploads, the kernel trusts them (like `write_kv`'s slots). The scheduler's `Request` can hold `SamplingParams` directly (it lives in `engine_core`) and the runner builds one `SampleRow` per running sequence with `step` = tokens generated so far (kept across preemption).
- The noise drops the batch row instead of using a sequence id: `curand_init(seed, 0, step * vocab + i)`. A request's tokens then depend only on (prompt, params, seed), so 16 concurrent requests match 16 sequential ones; requests without a seed get a random one.
- Greedy rows (temperature 0) are handled in the same kernel, so one launch serves a mixed batch; `greedy()` stays as the test oracle.
- `SamplingParams` field order is `{temperature, seed, top_k, top_p}` so existing `{t, seed}` initializers keep their meaning.
- Top-p keeps a whole tied group at the cut (the build plan's threshold rule); HF splits ties by sort order, which differs between CPU and CUDA. Superset of HF by a few tied tokens; see `technical_notes.md`.
- `top_p` must be in (0, 1]; HF also accepts 0 (keeps the top token in sort order). `top_k < 0` is rejected; 0 is off.

## Decisions in branch 19 (Claude's, open to Joe's change)

- `Request { id, prompt, max_new_tokens, params, output, state, preemptions, arrived, first_token, finished }` in `engine_core`; `max_new_tokens` on the request, not in `SamplingParams`. `validate(request, max_context)` is Part A.
- `Scheduler(SchedulerConfig{num_blocks, block_size, max_batch = 16, max_prefill_tokens = 2048, max_context = 2048, eos_id = -1})` owns the `BlockManager`; `submit`, `schedule() -> Step { prefill, seqs }`, `update(tokens)`, `take_finished()`, `request(id)`, `blocks()`, `num_waiting()`, `num_running()`.
- Admission is strict FIFO with `can_allocate(L)` (not `L + 1`), `L = prompt + output`; a step is all prefill (only the newly admitted) or all decode; preemption visits running sequences oldest first and preempts the newest until each can append; preempted requests go to the front of the queue in admission order; retirement in `update`.
- Constructor requires `max_prefill_tokens >= max_context` and `num_blocks - 1 >= ceil(max_context / block_size)` (liveness). Misuse throws `std::invalid_argument` with the strong guarantee.
- The M3 criterion is tested on the CPU with a fake model in this branch; the same check on the real model (16 concurrent vs 16 solo `generate` runs, with and without preemption) belongs to branch 20, once the paged runner exists.

## Decisions in branch 21 (confirmed by Joe on Oct 11)

- The summary goes in its own file, `<out>_summary.csv` (or `--summary`), one header and one row: the run's settings first (version, UTC date, GPU, build type, model, driver, max_batch, workload options), then the summary fields, `decode_weight_bytes` and `bandwidth_gb_s`. The per-request CSV stays a clean table.
- Requests ignore EOS by default, so every request generates exactly its output length; `--stop-at-eos` restores it.
- `--warmup` defaults to 3 untimed requests (seed + 1); without warmup the first TTFT was 600 ms instead of 25 ms.
- Request `i` samples with seed `seed + i`; the workload uses its own SplitMix64 so the same seed gives the same prompts on every standard library.
- Batch 1 (`run_one_at_a_time` over `generate`) until branch 20; `--max-batch`, `--graphs` and peak KV blocks come with the paged runner and CUDA Graphs. In an offline batch the TTFT includes waiting for earlier requests.
- Percentiles interpolate linearly (numpy's default); a one-token request has no TPOT (empty in the CSV) and is left out of the TPOT stats.
- `decode_weight_bytes` is the uploaded weights minus the embedding table unless tied (2,069,024,768 bytes, the build plan's 2.07 GB).

## Decisions in branch 20 (Claude's, open to Joe's change)

- `PagedRunner` is a new class; `ModelRunner` is untouched and stays the contiguous oracle for tests and the bench batch-1 path.
- Prefill attends through `paged_attention_decode` with one batch row per prompt token and `context_lens[t] = position + 1`; no paged prefill kernel yet (O(N^2) reads, as the build plan's MVP trick). Block tables are repeated per row (up to 2,048 x 128 ints).
- One host-to-device copy of a packed int vector per step (ids, positions, slots, context lengths, block tables); logits only for each sequence's last row, gathered with device-to-device row copies before the final RMSNorm and LM head.
- `build_paged_batch` lives in `engine_core` so it has CPU tests; it requires `blocks.length(id) == request.length()` for every sequence in the step.
- The "same text as solo" criterion holds exactly for greedy and under preemption; sampled concurrent vs solo can diverge late because batch 16 and batch 1 use different GEMM shapes (not batch invariant). The test asserts a common prefix of at least 16 tokens.
