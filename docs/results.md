# Results

What the engine measures against the five MVP criteria, against HuggingFace transformers and (when run) llama.cpp, and where a decode step spends its time. The blocks between `auto` markers are written by `scripts/make_results.py` from the files `scripts/reproduce.sh` produces in `results/`; everything else is hand-written and has to be revisited when a kernel lands (see the checklist at the end).

Regenerate everything with one command, plugged in and with no other GPU program running:

```bash
scripts/reproduce.sh            # about 10 minutes; --quick for a smoke run
```

## Environment

The commit is the code the numbers were measured on; the `results/` files and this page are committed right after it.

<!-- auto:environment -->
- Date: 2026-10-20T12:36:32Z
- Commit: 79c1f91
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Driver: 617.42
- CUDA: 13.3
- Build: Release
- Measured DRAM bandwidth (GB/s): 323.6
- Workload: 32 requests, 64 prompt tokens, 64 output tokens, greedy, seed 0, median of 3 run(s)
<!-- /auto:environment -->

## MVP scoreboard

<!-- auto:scoreboard -->
| # | Criterion | Target | Measured | |
|---|---|---|---|---|
| 1a | Teacher-forced vs HuggingFace | at least 99% argmax, error under 0.05 | 99.57% argmax over 1842 positions, mean abs logit error 0.0285 | met |
| 1b | Free-running greedy, first 64 tokens | exact on at least 18 of 20 | exact on 15 of 20; the other 5 diverge at a near tie (at most 2 BF16 ulps apart in HF's logits) | not met |
| 2 | Batch-1 decode bandwidth | at least 60% of 323.6 GB/s | 260 GB/s = 80% | met |
| 3 | Throughput at 16 concurrent requests | at least 5x batch 1 | 1386 vs 124 tokens/s = 11.1x | met |
<!-- /auto:scoreboard -->

Criterion 4 (each kernel has a before and after Nsight Compute number) and criterion 5 (reproducible from a clean clone) are tracked by the kernel table and `scripts/reproduce.sh` below rather than computed.

## Throughput and latency

![Throughput against concurrency](../results/plots/throughput_vs_batch.png)

![Median time per output token against concurrency](../results/plots/tpot_vs_batch.png)

<!-- auto:table -->
| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 114.3 | 8.66 | 8699.5 | 239.0 | 3 |
| engine | 2 | 209.1 | 9.25 | 4310.7 |  | 3 |
| engine | 4 | 426.4 | 8.96 | 2119.1 |  | 3 |
| engine | 8 | 757.3 | 9.61 | 1056.4 |  | 3 |
| engine | 16 | 1282.8 | 10.80 | 518.7 |  | 3 |
| engine + graphs | 1 | 124.4 | 7.96 | 7929.6 | 260.1 | 3 |
| engine + graphs | 4 | 468.4 | 8.07 | 1965.2 |  | 3 |
| engine + graphs | 16 | 1386.2 | 9.83 | 501.6 |  | 3 |
| HF transformers | 1 | 24.0 | 40.96 | 44404.9 | 50.5 | 1 |
| HF transformers | 4 | 89.6 | 45.80 | 9969.9 |  | 1 |
| HF transformers | 16 | 345.4 | 45.40 | 1590.1 |  | 1 |
<!-- /auto:table -->

The engine batches continuously (new requests join a running batch at any step, preemption on a full cache); the HuggingFace baseline uses static batches that run to the longest request, so the curves differ in more than kernel speed. `docs/baselines.md` lists what is and is not comparable.

## Batch-1 decode bandwidth

![Weights read per decode token against the measured peak](../results/plots/batch1_bandwidth.png)

Decoding one token reads every weight except all but one row of the embedding table, 2.07 GB, so the time per token cannot go below `2.07 GB / peak`. The peak used here is the DRAM bandwidth measured by `src/practice/bandwith_test.cu` on this GPU (323.6 GB/s, about 84% of the datasheet figure).

## Where a decode step goes

From `scripts/profile.sh nsys` on a batch-1 run (see `docs/profiling.md` for how to read it):

<!-- auto:profile -->
### Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 9343 us, median busy 7002 us, median idle 24.3% of the span, 39 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 13114 us, median busy 9899 us, median idle 24.4% of the span, 66 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3093.3 | 43.6 |
| down | 22.0 | 1705.0 | 24.0 |
| qkv | 22.0 | 743.3 | 10.5 |
| o_proj | 22.0 | 606.0 | 8.5 |
| lm_head | 1.0 | 394.5 | 5.6 |
| attention | 22.0 | 271.7 | 3.8 |
| add1_norm2 | 22.0 | 92.4 | 1.3 |
| add2_norm1 | 21.0 | 84.7 | 1.2 |
| rope | 22.0 | 49.6 | 0.7 |
| write_kv | 22.0 | 27.4 | 0.4 |
| swiglu | 22.0 | 20.3 | 0.3 |
| embedding | 1.0 | 2.8 | 0.0 |
| norm1 | 1.0 | 2.7 | 0.0 |
| final_norm | 1.0 | 2.7 | 0.0 |
| add2 | 1.0 | 0.8 | 0.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1650.2 | 73.6 | 90.1 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 77.8 | 13.8 | 4.3 |
| engine::fast::add_rmsnorm_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const __nv... | 11008 | 45.7 | 4.2 | 2.5 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 20.9 | 118.9 | 1.1 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 12.7 | 2.3 | 0.7 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.0 | 1.3 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.1 | 0.9 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.1 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.0 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.2 | 8.7 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.4 | 2.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.7 | 2.9 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 11.01 ms
- weights read per token / TPOT: 187.9 GB/s = 58.1% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.00 ms (64% of TPOT)
- weights / kernel busy time: 295.5 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile -->

The same step replayed from a CUDA Graph (`engine bench --graphs`, one graph per power-of-two batch size, kernels unchanged):

<!-- auto:profile_graphs -->
### Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 7091 us, median busy 7037 us, median idle 0.8% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 14199 us, median busy 10507 us, median idle 13.6% of the span, 46 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 224.0 | 7332.1 | 100.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1675.9 | 74.7 | 88.2 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 101.4 | 18.0 | 5.3 |
| engine::fast::add_rmsnorm_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const __nv... | 11008 | 55.6 | 5.1 | 2.9 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 24.2 | 137.6 | 1.3 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 15.1 | 2.7 | 0.8 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 8.2 | 1.5 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.5 | 1.0 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.1 | 46.8 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.0 | 44.9 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.8 | 11.1 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.7 | 3.4 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.9 | 3.3 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 7.90 ms
- weights read per token / TPOT: 261.9 GB/s = 80.9% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.04 ms (89% of TPOT)
- weights / kernel busy time: 294.0 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile_graphs -->

The launch gaps are gone (idle share of a decode step from about 10% to about 1%, gaps over 5 us from 15 per step to 0) and the kernels take the same time, so the saving is exactly the host-side launch time: a batch-1 token takes about 0.9 ms less (9.3 to 8.4 ms) and the weights bandwidth rises from about 223 to 247 GB/s, 76% of the measured peak (that was before paged attention v2 was wired in; the run above reaches 262 GB/s, 81%). What is left is the kernels themselves: the cuBLAS GEMV for gate/up and down owns about 68% of the step and runs at about 310 GB/s, and `paged_attention_decode_v2` owns 3.8% (v1 owned 11.5%). The eager profile above shows 24% idle time and 39 gaps per step against 10% and 15 in the previous run, and the HuggingFace rows are slower too, so the host was slower during this run (laptop clocks); the graph-replayed step is not affected by host speed.

## Kernels

Every kernel exists twice, a naive one that is obviously correct (the test oracle and the "before" number) and an optimized one tested against it. "In the forward pass" says whether `PagedRunner` calls the fast one; the end-to-end numbers above include only the kernels marked yes.

| Kernel | Optimized version | Tests | Nsight Compute before and after | In the forward pass |
| --- | --- | --- | --- | --- |
| cuBLAS GEMMs (QKV, output, gate/up, down, LM head) | none, cuBLAS as is | `tests/gpu/test_gemm.cpp` | batch-1 GEMVs run at about 310 GB/s in the profile above | yes |
| `paged_attention_decode` | v1 (one block per head and row, scores in shared memory) is the oracle and the "before" number; `paged_attention_decode_v2` (online softmax, 8 lanes per key, value sum spread over all threads, no context cap) is called by the forward pass | `tests/gpu/test_paged_attention.cpp`, `tests/gpu/test_paged_attention_v2.cpp` | v2 is 1.4x to 3.6x faster than v1 per launch (1 row, context 2,048: 600 to 165 us; 16 rows, context 512: 565 to 193 us) and runs a context of 8,192 that v1 cannot; Nsight Compute: 1,310 to 383 us at 1 row and context 2,048, details in `docs/profiling.md`, "Paged attention v2"; in the decode profile (32 prompt and 32 output tokens) attention went from 883 to 272 us per step (11.5% to 3.8% of kernel time) once v2 was wired in, `docs/profiling.md`, "Paged attention v2 in the forward pass" | yes, v2 |
| `write_kv` | one thread per element | `tests/gpu/test_write_kv.cpp` | not recorded | yes |
| add + RMSNorm | `fast::add_rmsnorm` (fused) | `tests/gpu/test_add_rmsnorm.cpp` | `docs/profiling.md`, "Fused add_rmsnorm" | yes, fused with the residual add |
| RoPE | `fast::rope` | `tests/gpu/test_rope_fast.cpp` | `docs/profiling.md`, "Fast RoPE" | yes |
| SwiGLU | `fast::swiglu` (vectorized, scalar below 5 tokens) | `tests/gpu/test_swiglu_fast.cpp` | `docs/profiling.md`, "Fast SwiGLU" and "Small batches" | yes |
| CUDA Graph decode | `DecodeGraphs` (one graph per power-of-two batch size up to the max batch, padded rows write nothing) | `tests/gpu/test_decode_graphs.cpp`, `tests/gpu/test_serve.cpp` | `docs/profiling.md`, "Decode CUDA Graphs"; launch gaps in the profile above | yes, `engine bench --graphs` |

## Reading these numbers

- "engine + graphs" is the same engine with `--graphs`; the scoreboard scores criteria 2 and 3 on the better of the two series, which is the graph one. HF and llama.cpp are never helped by it.
- The laptop GPU's clocks move between runs by 10 to 20%, so each configuration is run several times and the table shows the median of the runs.
- At batch 1 the engine's time per token is the sum of its kernels plus launch gaps and host work; the profile block above separates them.
- Both systems ignore EOS and generate exactly the requested number of tokens with greedy sampling, so the work per request is identical.
- The "Teacher-forced vs HuggingFace" and free-running rows run the contiguous `ModelRunner`, which keeps the naive kernels as the oracle. The paged runner that serves requests runs the fast kernels; its own HF check (`PagedRunnerTest.ConcurrentSequencesMatchHuggingFaceTeacherForced`, up to 16 sequences at once) reads 1166 of 1171 argmax matches (99.57%) with a mean absolute logit error of 0.0247.
- Free-running exact match is stricter than teacher forcing: a single BF16 near tie flips a token and the continuations then differ. The accepted rule (decided with the correctness suite) is that an unmatched prompt must diverge at a position where HuggingFace's own two logits are within 2 BF16 ulps.

## When a kernel lands

1. Wire it into `PagedRunner` (keep the naive kernel as the oracle in the tests).
2. Run `scripts/reproduce.sh` and read the scoreboard and the profile block.
3. Update its row in the Kernels table above and add its Nsight Compute before/after to `docs/profiling.md`.
4. Commit `results/` and `docs/results.md` together with the change, with the date, commit and GPU the script recorded.
