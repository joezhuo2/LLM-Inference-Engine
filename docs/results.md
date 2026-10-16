# Results

What the engine measures against the five MVP criteria, against HuggingFace transformers and (when run) llama.cpp, and where a decode step spends its time. The blocks between `auto` markers are written by `scripts/make_results.py` from the files `scripts/reproduce.sh` produces in `results/`; everything else is hand-written and has to be revisited when a kernel lands (see the checklist at the end).

Regenerate everything with one command, plugged in and with no other GPU program running:

```bash
scripts/reproduce.sh            # about 10 minutes; --quick for a smoke run
```

## Environment

<!-- auto:environment -->
<!-- /auto:environment -->

## MVP scoreboard

<!-- auto:scoreboard -->
<!-- /auto:scoreboard -->

Criterion 4 (each kernel has a before and after Nsight Compute number) and criterion 5 (reproducible from a clean clone) are tracked by the kernel table and `scripts/reproduce.sh` below rather than computed.

## Throughput and latency

![Throughput against concurrency](../results/plots/throughput_vs_batch.png)

![Median time per output token against concurrency](../results/plots/tpot_vs_batch.png)

<!-- auto:table -->
<!-- /auto:table -->

The engine batches continuously (new requests join a running batch at any step, preemption on a full cache); the HuggingFace baseline uses static batches that run to the longest request, so the curves differ in more than kernel speed. `docs/baselines.md` lists what is and is not comparable.

## Batch-1 decode bandwidth

![Weights read per decode token against the measured peak](../results/plots/batch1_bandwidth.png)

Decoding one token reads every weight except all but one row of the embedding table, 2.07 GB, so the time per token cannot go below `2.07 GB / peak`. The peak used here is the DRAM bandwidth measured by `src/practice/bandwith_test.cu` on this GPU (323.6 GB/s, about 84% of the datasheet figure).

## Where a decode step goes

From `scripts/profile.sh nsys` on a batch-1 run (see `docs/profiling.md` for how to read it):

<!-- auto:profile -->
<!-- /auto:profile -->

## Kernels

Every kernel exists twice, a naive one that is obviously correct (the test oracle and the "before" number) and an optimized one tested against it. "In the forward pass" says whether `PagedRunner` calls the fast one; the end-to-end numbers above include only the kernels marked yes.

| Kernel | Optimized version | Tests | Nsight Compute before and after | In the forward pass |
| --- | --- | --- | --- | --- |
| cuBLAS GEMMs (QKV, output, gate/up, down, LM head) | none, cuBLAS as is | `tests/gpu/test_gemm.cpp` | batch-1 GEMVs run at about 310 GB/s in the profile above | yes |
| `paged_attention_decode` | v1 (one block per head and row, scores in shared memory); online-softmax v2 not written | `tests/gpu/test_paged_attention.cpp` | not recorded | v1 |
| `write_kv` | one thread per element | `tests/gpu/test_write_kv.cpp` | not recorded | yes |
| add + RMSNorm | `fast::add_rmsnorm` (fused) | `tests/gpu/test_add_rmsnorm.cpp` | `docs/profiling.md`, "Fused add_rmsnorm" | no |
| RoPE | `fast::rope` | `tests/gpu/test_rope_fast.cpp` | `docs/profiling.md`, "Fast RoPE" | no |
| SwiGLU | `fast::swiglu` stub, tests waiting | `tests/todo/gpu/test_swiglu_fast.cpp` | not recorded | no |
| CUDA Graph decode | `DecodeGraphs` stub, runner prepared (`stage`, `run_staged`, `pad_decode_batch`) | `tests/todo/gpu/test_decode_graphs.cpp` | launch gaps in the profile above | no |

## Reading these numbers

- The laptop GPU's clocks move between runs by 10 to 20%, so each configuration is run several times and the table shows the median of the runs.
- At batch 1 the engine's time per token is the sum of its kernels plus launch gaps and host work; the profile block above separates them.
- Both systems ignore EOS and generate exactly the requested number of tokens with greedy sampling, so the work per request is identical.
- Free-running exact match is stricter than teacher forcing: a single BF16 near tie flips a token and the continuations then differ. The accepted rule (decided with the correctness suite) is that an unmatched prompt must diverge at a position where HuggingFace's own two logits are within 2 BF16 ulps.

## When a kernel lands

1. Wire it into `PagedRunner` (keep the naive kernel as the oracle in the tests).
2. Run `scripts/reproduce.sh` and read the scoreboard and the profile block.
3. Update its row in the Kernels table above and add its Nsight Compute before/after to `docs/profiling.md`.
4. Commit `results/` and `docs/results.md` together with the change, with the date, commit and GPU the script recorded.
