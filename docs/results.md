# Results

What the engine measures against the five MVP criteria, against HuggingFace transformers and llama.cpp, and where a decode step spends its time. The blocks between `auto` markers are written by `scripts/make_results.py` from the files `scripts/reproduce.sh` produces in `results/`; everything else is hand-written and has to be revisited when a kernel lands (see the checklist at the end).

Regenerate everything with one command, plugged in and with no other GPU program running:

```bash
scripts/reproduce.sh            # about 10 minutes; --quick for a smoke run
```

## Environment

The commit is the code the numbers were measured on; the `results/` files and this page are committed right after it.

<!-- auto:environment -->
- Date: 2026-10-25T09:33:51Z
- Commit: eafb664
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
| 2 | Batch-1 decode bandwidth | at least 60% of 323.6 GB/s | 265 GB/s = 82% | met |
| 3 | Throughput at 16 concurrent requests | at least 5x batch 1 | 1402 vs 128 tokens/s = 11.0x | met |
<!-- /auto:scoreboard -->

Criterion 4 (each kernel has a before and after Nsight Compute number) and criterion 5 (reproducible from a clean clone) are tracked by the kernel table and `scripts/reproduce.sh` below rather than computed.

## Throughput and latency

![Throughput against concurrency](../results/plots/throughput_vs_batch.png)

![Median time per output token against concurrency](../results/plots/tpot_vs_batch.png)

<!-- auto:table -->
| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 116.1 | 8.49 | 8519.7 | 243.7 | 3 |
| engine | 2 | 227.5 | 8.58 | 4125.8 |  | 3 |
| engine | 4 | 445.0 | 8.58 | 2082.6 |  | 3 |
| engine | 8 | 781.8 | 9.15 | 1082.8 |  | 3 |
| engine | 16 | 1263.7 | 11.02 | 522.2 |  | 3 |
| engine + graphs | 1 | 127.7 | 7.79 | 7732.0 | 265.4 | 3 |
| engine + graphs | 4 | 479.4 | 7.77 | 1937.1 |  | 3 |
| engine + graphs | 16 | 1402.2 | 9.68 | 498.1 |  | 3 |
| HF transformers | 1 | 27.8 | 33.64 | 32491.1 | 61.5 | 1 |
| HF transformers | 4 | 102.4 | 39.40 | 8975.4 |  | 1 |
| HF transformers | 16 | 396.8 | 39.44 | 1400.8 |  | 1 |
| llama.cpp | 1 | 110.1 | 8.95 | 9029.5 | 231.1 | 3 |
| llama.cpp | 2 | 193.8 | 9.84 | 4994.7 |  | 3 |
| llama.cpp | 4 | 395.5 | 9.61 | 2307.5 |  | 3 |
| llama.cpp | 8 | 780.2 | 9.49 | 1042.8 |  | 3 |
| llama.cpp | 16 | 1284.3 | 10.64 | 525.5 |  | 3 |
<!-- /auto:table -->

The engine batches continuously (new requests join a running batch at any step, preemption on a full cache); the HuggingFace and llama.cpp baselines use static batches that run to the longest request, so the curves differ in more than kernel speed. `docs/baselines.md` lists what is and is not comparable.

llama.cpp (release b11870, CUDA, the same BF16 weights as a GGUF, flash attention on `auto`, built and converted by `scripts/setup_llamacpp.sh`) is the serious baseline: a mature engine whose batch-1 decode is also bandwidth bound. Its rows come from `llama-batched-bench`, one static batch per size replayed over the 32 requests (`docs/baselines.md`). In this run the engine with CUDA Graphs is ahead at every size: TPOT 7.8 against 9.0 ms at batch 1 (265 against 231 GB/s of weights, 1.16x the tokens/s), 1.21x at batch 4 and 1.09x at batch 16. Eager, without graphs, the engine is ahead up to batch 4 and level from batch 8 (1264 against 1284 tokens/s at 16, inside the 10% run-to-run spread of llama.cpp's batch-16 runs, 1156 to 1315). The workload favors static batching (every request has the same length), and llama.cpp never runs a sampler in this tool while the engine samples on the GPU, so neither difference flatters the engine.

## Batch-1 decode bandwidth

![Weights read per decode token against the measured peak](../results/plots/batch1_bandwidth.png)

Decoding one token reads every weight except all but one row of the embedding table, 2.07 GB, so the time per token cannot go below `2.07 GB / peak`. The peak used here is the DRAM bandwidth measured by `src/practice/bandwith_test.cu` on this GPU (323.6 GB/s, about 84% of the datasheet figure).

## Where a decode step goes

From `scripts/profile.sh nsys` on a batch-1 run (see `docs/profiling.md` for how to read it):

<!-- auto:profile -->
### Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 8198 us, median busy 6918 us, median idle 11.8% of the span, 15 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 11352 us, median busy 10152 us, median idle 9.9% of the span, 46 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3161.5 | 43.9 |
| down | 22.0 | 1743.4 | 24.2 |
| qkv | 22.0 | 764.1 | 10.6 |
| o_proj | 22.0 | 621.8 | 8.6 |
| lm_head | 1.0 | 402.7 | 5.6 |
| attention | 22.0 | 317.1 | 4.4 |
| rope | 22.0 | 59.5 | 0.8 |
| add1_norm2 | 22.0 | 37.1 | 0.5 |
| add2_norm1 | 21.0 | 33.7 | 0.5 |
| write_kv | 22.0 | 30.9 | 0.4 |
| swiglu | 22.0 | 24.5 | 0.3 |
| embedding | 1.0 | 3.2 | 0.0 |
| norm1 | 1.0 | 3.2 | 0.0 |
| final_norm | 1.0 | 3.1 | 0.0 |
| add2 | 1.0 | 0.9 | 0.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1688.4 | 75.3 | 90.8 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 89.8 | 16.0 | 4.8 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.0 | 119.1 | 1.1 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 18.2 | 1.7 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 15.3 | 2.7 | 0.8 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.9 | 1.4 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 6.2 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.4 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.6 | 10.2 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.6 | 3.2 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.2 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 9.28 ms
- weights read per token / TPOT: 222.9 GB/s = 68.9% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.92 ms (75% of TPOT)
- weights / kernel busy time: 299.1 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile -->

The same step replayed from a CUDA Graph (`engine bench --graphs`, one graph per power-of-two batch size, kernels unchanged):

<!-- auto:profile_graphs -->
### Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 6983 us, median busy 6927 us, median idle 0.8% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 15989 us, median busy 10310 us, median idle 9.5% of the span, 62 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 224.0 | 7319.6 | 100.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1703.8 | 76.0 | 89.7 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 105.7 | 18.8 | 5.6 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 25.3 | 143.7 | 1.3 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 19.3 | 1.8 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 15.5 | 2.7 | 0.8 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 8.4 | 1.5 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.7 | 1.0 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.3 | 48.9 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.1 | 47.1 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 3.0 | 11.6 | 0.2 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.8 | 3.5 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.9 | 3.4 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 7.77 ms
- weights read per token / TPOT: 266.2 GB/s = 82.3% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.93 ms (89% of TPOT)
- weights / kernel busy time: 298.7 GB/s (what the kernels would reach with no launch gaps)
<!-- /auto:profile_graphs -->

The launch gaps are gone (idle share of a decode step from about 10% to about 1%, gaps over 5 us from 15 per step to 0) and the kernels take the same time, so the saving is exactly the host-side launch time: a batch-1 token takes about 0.9 ms less (9.3 to 8.4 ms) and the weights bandwidth rises from about 223 to 247 GB/s, 76% of the measured peak (that was before paged attention v2 was wired in; the run above reaches 262 GB/s, 81%). What is left is the kernels themselves: the cuBLAS GEMV for gate/up and down owns about 68% of the step and runs at about 310 GB/s, and `paged_attention_decode_v2` owns 3.8% (v1 owned 11.5%). The eager profile above shows 24% idle time and 39 gaps per step against 10% and 15 in the previous run, and the HuggingFace rows are slower too, so the host was slower during this run (laptop clocks); the graph-replayed step is not affected by host speed.

## Roofline

![Decode ops and whole decode steps against the measured bandwidth and compute ceilings](../results/plots/roofline.png)

From `scripts/roofline.py`: FLOPs and DRAM bytes of each GEMM and of attention are counted from `config.json` (weights, activations and the KV cache read once, nothing for caches), the op times come from the batch-1 profile above and a batch-16 profile (`results/decode_step_b16.md`), the whole-step points from the median TPOT of the table above. The ceilings are the measured DRAM bandwidth and the measured BF16 GEMM peak (`scripts/gemm_peak.cu`, `docs/hardware.md`).

<!-- auto:roofline -->
<!-- /auto:roofline -->

## Kernels

Every kernel exists twice, a naive one that is obviously correct (the test oracle and the "before" number) and an optimized one tested against it. "In the forward pass" says whether `PagedRunner` calls the fast one; the end-to-end numbers above include only the kernels marked yes.

| Kernel | Optimized version | Tests | Nsight Compute before and after | In the forward pass |
| --- | --- | --- | --- | --- |
| cuBLAS GEMMs (QKV, output, gate/up, down, LM head) | none, cuBLAS as is | `tests/gpu/test_gemm.cpp` | batch-1 GEMVs run at about 310 GB/s in the profile above | yes |
| `paged_attention_decode` | v1 (one block per head and row, scores in shared memory) is the oracle and the "before" number; `paged_attention_decode_v2` (online softmax, 8 lanes per key, value sum spread over all threads, no context cap) is called by the forward pass | `tests/gpu/test_paged_attention.cpp`, `tests/gpu/test_paged_attention_v2.cpp`, `tests/gpu/test_paged_attention_split.cpp` | v2 is 1.4x to 3.6x faster than v1 per launch (1 row, context 2,048: 600 to 165 us; 16 rows, context 512: 565 to 193 us) and runs a context of 8,192 that v1 cannot; Nsight Compute: 1,310 to 383 us at 1 row and context 2,048, details in `docs/profiling.md`, "Paged attention v2"; in the decode profile (32 prompt and 32 output tokens) attention went from 883 to 272 us per step (11.5% to 3.8% of kernel time) once v2 was wired in, `docs/profiling.md`, "Paged attention v2 in the forward pass"; `paged_attention_decode_split` (the context of a row cut across blocks, partials merged in split order) is 2.8x faster than v2 at 1 row and 2,048 tokens and takes the batch-1 TPOT at a 1,984-token prompt from 14.7 to 10.2 ms eager and from 14.0 to 9.1 ms with graphs, `docs/profiling.md`, "Split-context attention", "Split attention in the forward pass" and "Split attention in CUDA Graphs" | yes, v2, and split for 1 to 4 rows at 512 tokens or more (eager and graph replays) |
| `write_kv` | one thread per element | `tests/gpu/test_write_kv.cpp` | not recorded | yes |
| add + RMSNorm | `fast::add_rmsnorm` (fused; 128-bit loads, row kept in registers) | `tests/gpu/test_add_rmsnorm.cpp` | `docs/profiling.md`, "Fused add_rmsnorm" and "Tuning add_rmsnorm" (10.9 to 4.0 us at 1 token, 11.1 to 5.3 us at 64; 105 us less per decode step) | yes, fused with the residual add |
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
