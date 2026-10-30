# Branch 43 `runtime/graphs-split-attention` (PR summary)

Makes CUDA Graph replays use split attention, the follow-up proposed in `branch42_summary.md`. Joe chose "Claude implements it" (Part C, drafted by Claude for him to read). Three commits on eab6655 (#52), each gated with the GPU (debug 571/575/575, cpu 192):

1. 7770e78 `DecodeGraphs` keys its graphs by (bucket, split count). For each bucket the constructor walks contexts 0..`runner.max_context()`, and at the first context giving a new `attention_splits(bucket, heads, context)` it stages the dummy batch with that context and captures. `forward` picks the graph from `attention_splits(bucket, heads, padded.longest_context())`, the same value `run_staged` uses, so a replay is exactly the padded eager step. `covers` rejects a batch whose longest context exceeds the runner's max context (serve then runs eager instead of hitting a missing graph). `PagedRunner` gains `max_context()` and `config()`. TinyLlama at 2,048: 11 graphs (bucket 1: 1/2/4/8, bucket 2: 1/2/4, bucket 4: 1/2, 8 and 16: 1). At max context 128 nothing changes.
2. a5a7016 four GPU tests: graph count at 2,048; 1/2/3 long rows bit-identical to padded eager for 3 steps; a 1,018-token request crossing 1,024 (2 to 4 splits) bit-identical for 10 steps; `covers` at and past max context. Mutant "always replay the 1-split graph" fails both replay tests. Memcheck on all 12 DecodeGraphs tests: 0 errors, 0 leaks.
3. 3979f78 docs: `docs/profiling.md` "Split attention in CUDA Graphs (Branch 43)", the results Kernels row, the README (CUDA Graphs bullet; "split decode inside CUDA Graphs" dropped from the future list).

Numbers (TPOT p50, max batch 1, graphs): 1,984-token prompt 14.0 to 9.1 ms (1.5x), 1,024 tokens 10.0 to 8.3 ms; 64 and 512 unchanged. Graphs beat eager again at every length. The reproduce.sh workload never splits, so the measured blocks of `docs/results.md` are unchanged.

Subtle lines for Joe:
- `src/runtime/decode_graphs.cpp` constructor loop: the dummy keeps an all-zero block table while its context goes up to 2,048, so the capture warm-up reads cache block 0 repeatedly (read only, slot mapping -1 so nothing is written; the KV-cache-unchanged test still passes). The output of that warm-up is garbage and discarded.
- `forward` computes the split count before `stage()`; it must match what `run_staged` computed at capture, which holds because both call `attention_splits` on the same (rows = bucket, longest context) pair.
- Not measured: extra capture time at startup and device memory of 6 more executable graphs.
