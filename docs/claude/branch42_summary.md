# Branch 42 `runtime/wire-attention-split` (PR summary)

Wires `paged_attention_decode_split` (#51) into `PagedRunner`. Three commits on b597d93, each gated with the GPU (debug 570/571, cpu 192):

1. cde3aa1 `attention_splits(rows, heads, longest_context)` and `PagedBatch::longest_context()` in engine_core with CPU tests (Part B). Rule from `docs/profiling.md`: double from 1 while the grid stays at most 256 blocks (`kSplitAttentionBlocks`), the split count at most 8, and every split keeps at least 256 tokens of the longest context. 1 row: 2 splits from 512 tokens, 4 from 1,024, 8 from 2,048 (never reached with TinyLlama's 2,048 positions). 2/3/4 rows: 4/2/2. 5 or more rows, or a context under 512: 1 (v2).
2. e298faf `PagedRunner`: `stage()` records the longest context, `run_staged()` picks the split count from it and `t`, and calls v2 for 1 split or the split kernel otherwise. The runner owns a 66 KB F32 scratch buffer (`256 * (head_dim + 2)` floats), which bounds any grid the rule allows. New GPU test: 1,100- and 520-token requests decoded together (4 splits) match two contiguous runners (worst rel L2 0.031). A mutant that skips the split call fails it (1.79).
3. 3eb9f4f (docs) `docs/profiling.md` "Split attention in the forward pass (Branch 42)", the results Kernels row, the README bullet.

Numbers (TPOT p50, max batch 1, eager): prompt 1,984 tokens 14.7 to 10.2/10.9 ms; 1,024 tokens 10.6 to 9.3/9.7 ms; 64 and 512 unchanged. Graphs unchanged.

Design notes for Joe:
- The split count depends on the batch's longest context, known on the host in `stage()`, so prefill and short contexts keep v2 and the reproduce.sh workload is bit-for-bit what it was.
- CUDA Graphs: `DecodeGraphs` captures every bucket on a dummy batch with context 0, so the captured step runs v2 and replays never split. Eager is now faster than graphs from about 1,024 tokens. The fix is Part C (DecodeGraphs): capture one graph per (bucket, split count) and pick the graph from `attention_splits(bucket, heads, longest_context)` in `forward`. `run_staged` already uses whatever `stage()` recorded, so staging a batch with the target context before capture is enough. Only buckets 1, 2 and 4 can split (1: 2 or 4 splits, 2: 2 or 4, 4: 2), so 5 extra graphs with TinyLlama.
- Subtle lines: `src/runtime/paged_runner.cpp` where `stage()` sets `longest_context_` (after the event sync, so a run in flight is not affected, though it only reads the value at enqueue time anyway), and the scratch Tensor built with `splits` as dim 2, which must match the split count the kernel reads from `scratch.shape[2]`.
