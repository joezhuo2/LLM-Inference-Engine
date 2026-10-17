# Results

What the engine measures against the five MVP criteria, against HuggingFace transformers and (when run) llama.cpp, and where a decode step spends its time. The blocks between `auto` markers are written by `scripts/make_results.py` from the files `scripts/reproduce.sh` produces in `results/`; everything else is hand-written and has to be revisited when a kernel lands (see the checklist at the end).

Regenerate everything with one command, plugged in and with no other GPU program running:

```bash
scripts/reproduce.sh            # about 10 minutes; --quick for a smoke run
```

## Environment

The commit is the code the numbers were measured on; the `results/` files and this page are committed right after it.

<!-- auto:environment -->
- Date: 2026-10-17T07:54:43Z
- Commit: 6b170a5
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Driver: 616.92
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
| 2 | Batch-1 decode bandwidth | at least 60% of 323.6 GB/s | 244 GB/s = 75% | met |
| 3 | Throughput at 16 concurrent requests | at least 5x batch 1 | 1060 vs 117 tokens/s = 9.1x | met |
<!-- /auto:scoreboard -->

Criterion 4 (each kernel has a before and after Nsight Compute number) and criterion 5 (reproducible from a clean clone) are tracked by the kernel table and `scripts/reproduce.sh` below rather than computed.

## Throughput and latency

![Throughput against concurrency](../results/plots/throughput_vs_batch.png)

![Median time per output token against concurrency](../results/plots/tpot_vs_batch.png)

<!-- auto:table -->
| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 107.3 | 9.28 | 9189.4 | 222.9 | 3 |
| engine | 2 | 188.7 | 10.71 | 4670.7 |  | 3 |
| engine | 4 | 389.9 | 9.82 | 2341.9 |  | 3 |
| engine | 8 | 656.3 | 11.21 | 1251.9 |  | 3 |
| engine | 16 | 992.7 | 14.28 | 648.4 |  | 3 |
| engine + graphs | 1 | 117.0 | 8.49 | 8428.6 | 243.6 | 3 |
| engine + graphs | 4 | 422.4 | 8.95 | 2174.9 |  | 3 |
| engine + graphs | 16 | 1059.6 | 13.25 | 617.2 |  | 3 |
| HF transformers | 1 | 33.4 | 28.97 | 29320.9 | 71.4 | 1 |
| HF transformers | 4 | 130.2 | 30.54 | 6984.5 |  | 1 |
| HF transformers | 16 | 487.3 | 31.83 | 1167.3 |  | 1 |
<!-- /auto:table -->

The engine batches continuously (new requests join a running batch at any step, preemption on a full cache); the HuggingFace baseline uses static batches that run to the longest request, so the curves differ in more than kernel speed. `docs/baselines.md` lists what is and is not comparable.

## Batch-1 decode bandwidth

![Weights read per decode token against the measured peak](../results/plots/batch1_bandwidth.png)

Decoding one token reads every weight except all but one row of the embedding table, 2.07 GB, so the time per token cannot go below `2.07 GB / peak`. The peak used here is the DRAM bandwidth measured by `src/practice/bandwith_test.cu` on this GPU (323.6 GB/s, about 84% of the datasheet figure).

## Where a decode step goes

From `scripts/profile.sh nsys` on a batch-1 run (see `docs/profiling.md` for how to read it):

<!-- auto:profile -->
### Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 8782 us, median busy 7725 us, median idle 11.4% of the span, 18 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 13411 us, median busy 11564 us, median idle 12.4% of the span, 59 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3118.8 | 40.0 |
| down | 22.0 | 1715.8 | 22.0 |
| attention | 22.0 | 880.4 | 11.3 |
| qkv | 22.0 | 750.6 | 9.6 |
| o_proj | 22.0 | 612.7 | 7.8 |
| lm_head | 1.0 | 395.6 | 5.1 |
| rope | 22.0 | 96.7 | 1.2 |
| norm2 | 22.0 | 71.6 | 0.9 |
| norm1 | 22.0 | 67.4 | 0.9 |
| write_kv | 22.0 | 29.7 | 0.4 |
| swiglu | 22.0 | 22.8 | 0.3 |
| add1 | 22.0 | 19.3 | 0.2 |
| add2 | 22.0 | 18.7 | 0.2 |
| embedding | 1.0 | 3.1 | 0.0 |
| final_norm | 1.0 | 3.0 | 0.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1663.2 | 74.1 | 82.5 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 236.2 | 41.9 | 11.7 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 36.6 | 3.2 | 1.8 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 24.9 | 4.4 | 1.2 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.0 | 119.1 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 10.0 | 0.9 | 0.5 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.6 | 1.4 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.4 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.1 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.2 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.5 | 9.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 9.53 ms
- weights read per token / TPOT: 217.1 GB/s = 67.1% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.73 ms (81% of TPOT)
- weights / kernel busy time: 267.8 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile -->

The same step replayed from a CUDA Graph (`engine bench --graphs`, one graph per power-of-two batch size, kernels unchanged):

<!-- auto:profile_graphs -->
### Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 7923 us, median busy 7854 us, median idle 0.9% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 12683 us, median busy 11595 us, median idle 8.5% of the span, 48 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 267.0 | 8049.3 | 100.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1702.0 | 75.9 | 81.8 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 258.7 | 45.9 | 12.4 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 38.7 | 3.4 | 1.9 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 25.5 | 4.5 | 1.2 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.7 | 123.5 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 9.2 | 0.8 | 0.4 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.8 | 1.4 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.1 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.6 | 41.2 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.5 | 39.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.7 | 10.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.2 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 8.55 ms
- weights read per token / TPOT: 242.0 GB/s = 74.8% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.85 ms (92% of TPOT)
- weights / kernel busy time: 263.4 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile_graphs -->

The launch gaps are gone (idle share of a decode step from about 11% to about 1%, gaps over 5 us from 17 or 18 per step to 0) and the kernels take the same time, so the saving is exactly the host-side launch time: a batch-1 token takes about 0.8 ms less (9.3 to 8.5 ms) and the weights bandwidth rises from about 223 to 244 GB/s, 75% of the measured peak. What is left is the kernels themselves: the cuBLAS GEMV for gate/up and down owns about 62% of the step and runs at about 310 GB/s, and `paged_attention_decode` v1 owns 11%.

## Kernels

Every kernel exists twice, a naive one that is obviously correct (the test oracle and the "before" number) and an optimized one tested against it. "In the forward pass" says whether `PagedRunner` calls the fast one; the end-to-end numbers above include only the kernels marked yes.

| Kernel | Optimized version | Tests | Nsight Compute before and after | In the forward pass |
| --- | --- | --- | --- | --- |
| cuBLAS GEMMs (QKV, output, gate/up, down, LM head) | none, cuBLAS as is | `tests/gpu/test_gemm.cpp` | batch-1 GEMVs run at about 310 GB/s in the profile above | yes |
| `paged_attention_decode` | v1 (one block per head and row, scores in shared memory); online-softmax v2 not written | `tests/gpu/test_paged_attention.cpp` | not recorded | v1 |
| `write_kv` | one thread per element | `tests/gpu/test_write_kv.cpp` | not recorded | yes |
| add + RMSNorm | `fast::add_rmsnorm` (fused) | `tests/gpu/test_add_rmsnorm.cpp` | `docs/profiling.md`, "Fused add_rmsnorm" | no |
| RoPE | `fast::rope` | `tests/gpu/test_rope_fast.cpp` | `docs/profiling.md`, "Fast RoPE" | no |
| SwiGLU | `fast::swiglu` (vectorized, scalar below 5 tokens) | `tests/gpu/test_swiglu_fast.cpp` | `docs/profiling.md`, "Fast SwiGLU" and "Small batches" | no |
| CUDA Graph decode | `DecodeGraphs` (one graph per power-of-two batch size up to the max batch, padded rows write nothing) | `tests/gpu/test_decode_graphs.cpp`, `tests/gpu/test_serve.cpp` | `docs/profiling.md`, "Decode CUDA Graphs"; launch gaps in the profile above | yes, `engine bench --graphs` |

## Reading these numbers

- "engine + graphs" is the same engine with `--graphs`; the scoreboard scores criteria 2 and 3 on the better of the two series, which is the graph one. HF and llama.cpp are never helped by it.
- The laptop GPU's clocks move between runs by 10 to 20%, so each configuration is run several times and the table shows the median of the runs.
- At batch 1 the engine's time per token is the sum of its kernels plus launch gaps and host work; the profile block above separates them.
- Both systems ignore EOS and generate exactly the requested number of tokens with greedy sampling, so the work per request is identical.
- Free-running exact match is stricter than teacher forcing: a single BF16 near tie flips a token and the continuations then differ. The accepted rule (decided with the correctness suite) is that an unmatched prompt must diverge at a position where HuggingFace's own two logits are within 2 BF16 ulps.

## When a kernel lands

1. Wire it into `PagedRunner` (keep the naive kernel as the oracle in the tests).
2. Run `scripts/reproduce.sh` and read the scoreboard and the profile block.
3. Update its row in the Kernels table above and add its Nsight Compute before/after to `docs/profiling.md`.
4. Commit `results/` and `docs/results.md` together with the change, with the date, commit and GPU the script recorded.
