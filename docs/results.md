# Results

What the engine measures against the five MVP criteria, against HuggingFace transformers and (when run) llama.cpp, and where a decode step spends its time. The blocks between `auto` markers are written by `scripts/make_results.py` from the files `scripts/reproduce.sh` produces in `results/`; everything else is hand-written and has to be revisited when a kernel lands (see the checklist at the end).

Regenerate everything with one command, plugged in and with no other GPU program running:

```bash
scripts/reproduce.sh            # about 10 minutes; --quick for a smoke run
```

## Environment

The commit is the code the numbers were measured on; the `results/` files and this page are committed right after it.

<!-- auto:environment -->
- Date: 2026-10-16T01:42:30Z
- Commit: d5debf7
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
| 2 | Batch-1 decode bandwidth | at least 60% of 323.6 GB/s | 225 GB/s = 69% | met |
| 3 | Throughput at 16 concurrent requests | at least 5x batch 1 | 1014 vs 108 tokens/s = 9.4x | met |
<!-- /auto:scoreboard -->

Criterion 4 (each kernel has a before and after Nsight Compute number) and criterion 5 (reproducible from a clean clone) are tracked by the kernel table and `scripts/reproduce.sh` below rather than computed.

## Throughput and latency

![Throughput against concurrency](../results/plots/throughput_vs_batch.png)

![Median time per output token against concurrency](../results/plots/tpot_vs_batch.png)

<!-- auto:table -->
| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 107.6 | 9.21 | 9199.0 | 224.5 | 3 |
| engine | 2 | 191.6 | 10.05 | 4602.6 |  | 3 |
| engine | 4 | 399.4 | 9.57 | 2287.7 |  | 3 |
| engine | 8 | 665.2 | 11.07 | 1228.5 |  | 3 |
| engine | 16 | 1014.2 | 13.97 | 635.7 |  | 3 |
| HF transformers | 1 | 33.7 | 28.80 | 28728.1 | 71.8 | 1 |
| HF transformers | 4 | 115.6 | 32.73 | 7000.5 |  | 1 |
| HF transformers | 16 | 427.7 | 36.51 | 1183.1 |  | 1 |
<!-- /auto:table -->

The engine batches continuously (new requests join a running batch at any step, preemption on a full cache); the HuggingFace baseline uses static batches that run to the longest request, so the curves differ in more than kernel speed. `docs/baselines.md` lists what is and is not comparable.

## Batch-1 decode bandwidth

![Weights read per decode token against the measured peak](../results/plots/batch1_bandwidth.png)

Decoding one token reads every weight except all but one row of the embedding table, 2.07 GB, so the time per token cannot go below `2.07 GB / peak`. The peak used here is the DRAM bandwidth measured by `src/practice/bandwith_test.cu` on this GPU (323.6 GB/s, about 84% of the datasheet figure).

## Where a decode step goes

From `scripts/profile.sh nsys` on a batch-1 run (see `docs/profiling.md` for how to read it):

<!-- auto:profile -->
### Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 8703 us, median busy 7648 us, median idle 11.2% of the span, 17 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 12997 us, median busy 11685 us, median idle 8.3% of the span, 50 gaps over 5 us per step

### GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3057.6 | 39.9 |
| down | 22.0 | 1686.4 | 22.0 |
| attention | 22.0 | 868.5 | 11.3 |
| qkv | 22.0 | 737.5 | 9.6 |
| o_proj | 22.0 | 602.3 | 7.9 |
| lm_head | 1.0 | 388.2 | 5.1 |
| rope | 22.0 | 96.7 | 1.3 |
| norm2 | 22.0 | 71.1 | 0.9 |
| norm1 | 22.0 | 66.9 | 0.9 |
| write_kv | 22.0 | 29.4 | 0.4 |
| swiglu | 22.0 | 22.7 | 0.3 |
| add1 | 22.0 | 19.3 | 0.3 |
| add2 | 22.0 | 18.7 | 0.2 |
| embedding | 1.0 | 3.0 | 0.0 |
| final_norm | 1.0 | 3.0 | 0.0 |

### Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1632.6 | 72.8 | 82.4 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 233.6 | 41.5 | 11.8 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 36.4 | 3.2 | 1.8 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 24.9 | 4.4 | 1.3 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 20.6 | 117.2 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 10.0 | 0.9 | 0.5 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.5 | 1.3 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.4 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.4 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.2 | 36.9 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.5 | 9.6 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

### Decode bandwidth

- max_batch 1, median TPOT 9.37 ms
- weights read per token / TPOT: 220.9 GB/s = 68.3% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.65 ms (82% of TPOT)
- weights / kernel busy time: 270.5 GB/s (what the kernels would reach with no launch gaps)
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
