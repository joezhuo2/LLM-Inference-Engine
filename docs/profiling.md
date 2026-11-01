# Profiling

How to see where a decode step spends its time. Everything runs on the `release` preset, plugged in and in maximum performance mode, with `engine bench --profile`, which calls `cudaProfilerStart` and `cudaProfilerStop` around the timed requests so the profilers skip the weight upload, the cuBLAS warm-up and the untimed `--warmup` requests.

## Timeline and report (Nsight Systems)

```bash
scripts/profile.sh nsys b1
```

This runs `engine bench --max-batch 1 --num-requests 4 --prompt-len 32 --output-len 32` under `nsys profile`, writes `results/profile/b1.nsys-rep` (open it in `nsys-ui`) and prints `scripts/profile_report.py`'s tables, also saved to `results/profile/b1.md`. Anything after the tag replaces the default bench options, for example `scripts/profile.sh nsys b16 --max-batch 16 --num-requests 32 --prompt-len 128 --output-len 64`. Use `--max-batch N` (any N, including 1): the batched driver is the one with NVTX ranges.

The NVTX ranges in the timeline are:

- per serving step: `step` containing `schedule`, `build_batch`, `forward_decode` or `forward_prefill`, `sample`, `read_tokens` and `update`;
- inside the forward: `stage`, `embedding`, then `layer 0` to `layer 21` each with `norm1`, `qkv`, `rope`, `write_kv`, `attention`, `o_proj`, `add1`, `norm2`, `gate_up`, `swiglu`, `down`, `add2`, then `final_norm` and `lm_head`.

A range is drawn on the host thread over the time the CPU spent launching its kernels. The GPU runs those kernels later, so in the GPU rows the kernel boxes are what to time; `profile_report.py` follows each kernel back to the op that launched it.

## What the report says

- **Launch gaps per forward step.** The span (first kernel start to last kernel end), the busy time (kernels overlapping count once), the idle share and the number of gaps over 5 us. This is the M4 criterion "nsys timeline shows no launch gaps": with CUDA Graphs the idle share and the gap count should fall to near zero.
- **GPU time per op in a decode step.** Which of the 12 ops per layer owns the time; the optimized kernels are measured against the same rows.
- **Top kernels.** The same time by kernel name, which separates cuBLAS kernels from the engine's own.
- **Decode bandwidth** (with `--summary`, which `profile.sh` passes). The weights a decode step reads (2.07 GB, `--weight-bytes`) divided by the median TPOT, as a share of the measured peak (`--peak-gb-s`, default 323.6) against the 60% target, plus the same bytes divided by the GPU busy time, which is what the kernels alone would reach without launch gaps and host work. The gap between the two numbers is what CUDA Graphs and a lighter host loop can recover; the gap between the second number and the peak is what the kernels themselves can recover.

## One kernel in depth (Nsight Compute)

```bash
scripts/profile.sh ncu attn paged_attention --num-requests 1 --prompt-len 32 --output-len 8 --max-batch 1
```

`ncu` profiles the first `NCU_COUNT` (default 4) launches whose name matches the regex, with section set `NCU_SET` (default `full`), and writes `results/profile/attn_ncu.ncu-rep`. Open it with `ncu-ui`, or `ncu --import results/profile/attn_ncu.ncu-rep --page details` for text. Counter replay serializes and slows the program by orders of magnitude, so the TTFT and TPOT the bench prints during an `ncu` run mean nothing; keep the request count and output length small. Take a "before" report on the naive kernel and an "after" on the optimized one with the same arguments, and compare DRAM throughput, achieved occupancy and the stall reasons.

## Reading the numbers honestly

- Run each configuration several times and quote the median: the laptop GPU's clocks move between runs (the same batch-1 decode measured 8.4 to 10.4 ms per token on different runs).
- The bench's TPOT includes the host side of a step (scheduling, batch building, the token read-back). The report's busy time does not. Compare like with like.
- Only the capture window is profiled; changing `--warmup` or the prompt lengths changes what the window contains.

## Decode CUDA Graphs (Branch 34)

Measured on the RTX 5060 Laptop GPU, release build, `scripts/profile.sh nsys` on a batch-1 run of 4 requests with 64 prompt and 64 output tokens, with and without `--graphs`, plus `engine bench` on 32 requests of the same shape (median of 3 runs, plugged in).

| decode step at batch 1 | eager | graph replay |
| --- | --- | --- |
| kernels per step | 267 | 267 |
| median span | 9046 us | 7913 us |
| median busy | 7938 us | 7844 us |
| median idle | 11.3% | 0.9% |
| gaps over 5 us per step | 17 | 0 |
| median TPOT | 9.49 ms | 8.51 ms |
| weights bandwidth | 218 GB/s (67%) | 243 GB/s (75%) |

| `engine bench`, output tokens/s | eager | graphs | gain |
| --- | --- | --- | --- |
| max batch 1 | 105.1 | 116.8 | +11% |
| max batch 4 | 391.3 | 418.3 | +7% |
| max batch 16 | 997.5 | 1057.4 | +6% |

Reading it. The kernels are the same and take the same time (busy 7.9 ms against 7.8 ms), so the whole gain is in the time between kernels. That time is not the GPU waiting for the CPU: in the eager profile the `cudaLaunchKernel` calls (13 us each under the profiler) run a median 1.1 ms ahead of the kernels they launch, and at only 0.2% of the boundaries between two kernels was the next launch call still running when the previous kernel ended. The idle time is a gap of a few microseconds before every kernel (median 3.3 us, 10% to 90% of the gaps between 2.2 and 4.3 us, over the 266 boundaries of each of the 252 decode steps), about 1.0 ms per step; replayed from a graph the same gaps are 0.28 us (0.25 to 0.32). So the graph pays off by removing a per-launch cost on the GPU side, not by taking the CPU off the critical path, and fusing kernels (fewer boundaries) shrinks the same cost. The saving per decode step does not depend on the batch: an eager batch-16 step (a later profile, `results/decode_step_b16.md`, 224 kernels) has a 2.8 us median gap and 1.1 ms of idle like batch 1, and `docs/results.md` shows 0.71, 0.82 and 1.00 ms saved per token at max batch 1, 4 and 16. The throughput gain still shrinks with the batch because the decode step gets longer and because prefill steps, which are not graphed and keep their gaps (about 9% idle), are a larger share of a batched run: the batch-16 profile has one 113 ms prefill of 16 prompts against 63 decode steps of 9.7 ms. The gap numbers come from the Nsight Systems sqlite exports of these profiles, with kernels joined to their launch calls by `correlationId` as `scripts/profile_report.py` does; `web/scripts/step-timeline.py` writes the same quantiles for the later profiles into `web/src/content/kernels/cuda-graphs-trace.json`.

The profile needs `--cuda-graph-trace=node` (already in `profile.sh`) so the kernels inside a graph launch show up in the timeline; without it the whole replay is one box. Compute-sanitizer memcheck on all 8 `DecodeGraphsTest` tests reports 0 errors and 0 bytes leaked. Reproduce with `scripts/profile.sh nsys decode_eager --max-batch 1 --num-requests 4 --prompt-len 64 --output-len 64` and the same with `--graphs`.

## Fast kernels in the forward pass (Branch 36)

`PagedRunner` now runs `fast::rope`, `fast::swiglu` and `fast::add_rmsnorm`: each residual add is fused with the norm after it (`add1_norm2` in every layer, `add2_norm1` into the next layer's input norm), which takes a decode step from 267 to 224 kernels. Measured with `scripts/reproduce.sh` on the RTX 5060 Laptop GPU, release build, a batch-1 run of 4 requests with 64 prompt and 64 output tokens for the profile and 32 requests of the same shape (median of 3) for the throughput, before (commit 6b170a5) and after (commit e07c022).

| decode step at batch 1 | naive kernels, eager | fast kernels, eager | naive kernels, graphs | fast kernels, graphs |
| --- | --- | --- | --- | --- |
| kernels per step | 267 | 224 | 267 | 224 |
| median busy | 7725 us | 7635 us | 7854 us | 7587 us |
| median idle | 11.4% | 10.0% | 0.9% | 0.7% |
| gaps over 5 us per step | 18 | 15 | 0 | 0 |
| median TPOT | 9.53 ms | 9.30 ms | 8.49 ms | 8.37 ms |

| `engine bench`, output tokens/s | naive eager | fast eager | naive graphs | fast graphs |
| --- | --- | --- | --- | --- |
| max batch 1 | 107.3 | 107.6 | 117.0 | 117.9 |
| max batch 4 | 389.9 | 396.0 | 422.4 | 426.5 |
| max batch 16 | 992.7 | 1011.9 | 1059.6 | 1074.8 |

Reading it. The gain is small, 1% to 2% of throughput, which is about the run-to-run spread of the laptop's clocks, so treat the direction as real and the size as approximate. Per op in the eager profile, `rope` is the clear win (96.7 us to 56.1 us per step) and `swiglu` is unchanged (it was already 23 us); the fused add and norm is not faster than the pair it replaces in place: `add1_norm2` takes 4.8 us per launch (106.3 us over 22 launches) against 4.1 us for the naive `add1` plus `norm2` (19.3 us and 71.6 us over 22 launches), and `add2_norm1` takes 4.5 us per launch against 4.1. The Nsight Compute comparison in "Fused add_rmsnorm" measured 11.4 to 11.8 us for the fused kernel against 11.9 to 14.5 us for the naive pair at 1 to 64 rows, from isolated launches; inside the decode step the naive pair takes 4.1 us per layer, so the fusion saves launches (43 per step, worth about 0.2 ms of eager TPOT) and DRAM bytes but not kernel time. Each of these kernels is under 2% of a step; the step is 62% gate/up and down GEMVs, so the fast kernels cannot move the total by more than a few percent whatever they do.

The logits move by rounding because `fast::add_rmsnorm` sums the squares in another order than `naive::rmsnorm`: the paged HF teacher-forced check reads 1166 of 1171 (99.57%, mean absolute logit error 0.0247) against 99.66% and 0.025 before. In the 16-request greedy test one request now leaves its solo run at token 27, where two logits tie exactly in BF16 (15.625 each, the third best is 14.56), so the serve tests accept a divergence at an exact or two-ulp tie and nowhere else.

## Fused add_rmsnorm (Branch 30)

Measured with Nsight Compute on the RTX 5060 Laptop GPU (release build, hidden 2048 as in TinyLlama, all values constant so only the timing matters, default clock control, first three launches after a warm-up of three each). `naive` is `naive::add` plus `naive::rmsnorm` (two kernels), `fast` is `fast::add_rmsnorm` (one kernel).

| tokens | naive add (us) | naive rmsnorm (us) | naive total (us) | fast (us) | time ratio | naive DRAM (KB) | fast DRAM (KB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.9 | 8.1 | 12.0 | 11.8 | 0.98 | 29.2 | 23.6 |
| 16 | 4.3 | 7.6 | 11.9 | 11.4 | 0.96 | 213.5 | 142.3 |
| 64 | 6.6 | 7.9 | 14.5 | 11.5 | 0.79 | 803.6 | 535.6 |

Reading it. The fused kernel writes the sum and the norm once and reads the sum back from L2 instead of DRAM, so it moves about 2/3 of the DRAM bytes. In GPU time it only wins by 2% to 21%, because both versions are latency bound, not bandwidth bound: at 1 to 64 rows the kernel takes about 11.5 us regardless of rows, while 64 rows of 4 KB are 0.5 MB, which is under 2 us of DRAM time at 323.6 GB/s. The time goes to the two dependent passes over the row, the block reduction and its barriers, with eight scalar 2-byte loads per thread. The real saving in a decode step is one launch less per norm (44 launches per step with 22 layers), which these per-kernel numbers do not show; that shows in the nsys launch-gap table once the kernel is wired into `PagedRunner` (branch 23). Correctness is the part of this kernel that is finished; the measured headroom is in the load width and the number of passes, not in the arithmetic.

To reproduce, launch each kernel in a small throwaway program (the three calls in a loop, same tensors) and run `ncu --kernel-name regex:"add_kernel|rmsnorm_kernel|add_rmsnorm_kernel" --launch-skip 6 --launch-count 3 --metrics gpu__time_duration.sum,dram__bytes.sum <program> <tokens>`.

## Fast RoPE (Branch 31)

Measured with Nsight Compute on the RTX 5060 Laptop GPU (release build, TinyLlama's layout of 32 query heads, 4 key/value heads and head_dim 64, positions `3 * i`, default clock control, one launch after a warm-up of ten). `naive` is `naive::rope`. `first draft` is the commit before the parallelization, with one thread per head and a serial loop over the 32 pairs; `fast` is the final kernel, one block per token with threads over pairs and head groups. All three are bit-identical (`FastRope.*`, 9 tests).

| tokens | naive (us) | first draft (us) | fast (us) | fast / naive | DRAM, fast (KB) |
| --- | --- | --- | --- | --- | --- |
| 1 | 9.0 | 21.6 | 6.5 | 0.72 | 15.4 |
| 16 | 9.0 | 22.0 | 7.0 | 0.78 | 84.5 |
| 64 | 10.5 | 21.1 | 6.9 | 0.66 | 306.2 |
| 512 | 36.1 | 21.5 | 17.9 | 0.50 | 2,370 |

Reading it. RoPE reads and writes only the Q and K slices of each row, so at 512 tokens the DRAM traffic is about 2.4 MB and the kernel is nowhere near the bandwidth limit: below 64 tokens it is latency bound at 6 to 7 us, which is mostly launch and one dependent chain of loads, sincos and stores. The first draft was a flat 21 us at every size because each thread walked all 36 heads one after another, a dependent global round trip each; it only beat naive once there were 512 tokens to hide it. The final kernel computes `cosf` and `sinf` once per `(token, pair)` and spreads the heads over 8 thread groups, which is 1.4x to 1.5x faster than naive at decode sizes and 2x at 512 tokens. In a decode step it saves about 2 to 3 us of the roughly 9 ms step, so the benefit is small in absolute terms; the point of the exercise is that it is no longer slower than the kernel it replaces.

To reproduce, launch `naive::rope` and `fast::rope` in a small throwaway program (same `qkv` and positions, 10 warm-up launches each, one program run per token count) linked against `libengine_kernels.a` and `libengine_core.a`, and run `ncu --kernel-name regex:rope_kernel --launch-skip <n> --launch-count 1 --metrics gpu__time_duration.sum,dram__bytes.sum <program>`.

## Fast SwiGLU (Branch 32)

Measured with Nsight Compute on the RTX 5060 Laptop GPU (release build, TinyLlama's `ff` of 5632, bf16 input of shape `[tokens, 11264]`, default clock control, median of three runs of one launch after a warm-up of thirty). `naive` is `naive::swiglu`. `first draft` is the commit before the vectorization, with one thread per output element on a 2D grid; `fast` is the final kernel, which loads 8 gate and 8 up values per thread with 16-byte loads (the scalar kernel is the fallback for odd `ff` or misaligned pointers and is not measured here). All three are bit-identical (`FastSwiglu.*` and `SwigluShapeTest.*`, 25 tests).

| tokens | naive (us) | first draft (us) | fast (us) | fast / naive | DRAM, fast (KB) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.8 | 3.1 | 3.8 | 1.00 | 32 |
| 16 | 6.1 | 5.1 | 4.8 | 0.78 | 362 |
| 64 | 12.9 | 11.1 | 8.1 | 0.62 | 1,418 |
| 512 | 78.4 | 68.3 | 38.1 | 0.49 | 11,274 |
| 4096 | 650.5 | 576.4 | 358.2 | 0.55 | 115,963 |

Reading it. SwiGLU reads two values and writes one per output element and does a few dozen FLOPs on them, so it is memory bound once there is enough work to fill the GPU: at 4096 tokens the fast kernel moves about 119 MB in 358 us, roughly 330 GB/s, which is the measured peak of 323.6 GB/s used in the bandwidth numbers above, while naive reaches about 200 GB/s. From 64 tokens up the vectorized kernel is 1.6x to 2x faster than naive. At 1 token (the decode case) there is no gain: 5632 outputs make only 704 vectors, which is 3 blocks of 256 threads on a GPU with 26 SMs, and each thread runs 8 dependent `expf` and divide chains, so the kernel is latency bound at about 3.8 us, 0.7 us slower than the first draft, whose 22 blocks spread the same chains wider. A decode step runs the kernel 22 times, so the difference is about 15 us of a roughly 9 ms step. Not wired into `PagedRunner` yet.

To reproduce, launch `naive::swiglu`, the first draft (the `swiglu_fast.cu` of the first implementation commit, renamed to another namespace) and `fast::swiglu` in a small throwaway program (same `gate_up`, 10 warm-up launches each, one program run per token count) linked against `libengine_kernels.a` and `libengine_core.a`, and run `ncu --kernel-name regex:swiglu_ --launch-skip 30 --launch-count 3 --metrics gpu__time_duration.sum,dram__bytes.sum <program> <tokens>`.

### Small batches (Branch 33)

The 1-token row above showed the vector kernel no faster than naive, so `fast::swiglu` now takes the scalar kernel (256 threads, one element each) when a launch has fewer than `sm_count * 256 / 2` vectors, which is 4 tokens or fewer at `ff` 5632 on this 26-SM GPU, and the vector kernel from 5 tokens up. Same method as above, five runs per row, `previous` is the vector-only kernel of Branch 32 and `fast` the dispatching kernel. The laptop clocks differ between sessions (naive at 1 token read 3.8 us here and 2.8 us in an earlier session), so compare the ratios within a row, not the absolute times with the table above.

| tokens | naive (us) | previous (us) | fast (us) | fast / naive | kernel used |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.78 | 3.87 | 3.26 | 0.86 | scalar |
| 4 | 4.03 | 3.94 | 3.39 | 0.84 | scalar |
| 5 | 4.03 | 3.94 | 3.94 | 0.98 | vector |
| 8 | 5.06 | 4.16 | 4.13 | 0.82 | vector |
| 512 | 78.6 | 37.8 | 37.8 | 0.48 | vector |

Reading it. At decode sizes the work is a few thousand elements, so what matters is how many blocks and threads share it: 22 blocks of scalar threads each do one `expf` chain, while the vector kernel puts 8 chains in each of 704 threads (3 blocks). Below 5 tokens the scalar kernel is 14% to 16% faster than the vector kernel in the same run and 14% to 16% faster than naive; the two kernels are equal around 5 to 8 tokens and the vector kernel wins from there. An earlier session also measured the first draft (the same scalar kernel on a 2D grid) at 2.14 us against 2.82 us for naive at 1 token, so about 0.75x of naive is the floor this size can reach. In a decode step that is about 0.5 us per layer, roughly 11 us of a 9 ms step: real but small, and the fused GEMM epilogue (not in this branch) is where the kernel would disappear altogether.

## Paged attention v2 (Branch 37)

`paged_attention_decode_v2` against v1 (`paged_attention_decode`), RTX 5060 Laptop GPU, release build, 32 heads, 4 KV heads, head_dim 64, block size 16, every row at the full context. Time per launch in us from a loop of 50 launches after 3 warm-up launches (`scripts/paged_attention_bench.cu`); v1 cannot run a context of 8,192 because its scores live in shared memory (shown as n/a).

| rows | context | v1 (us) | v2 (us) | v1 / v2 |
| --- | --- | --- | --- | --- |
| 1 | 128 | 41.7 | 30.1 | 1.4 |
| 1 | 512 | 154.1 | 43.1 | 3.6 |
| 1 | 2,048 | 600.2 | 165.0 | 3.6 |
| 1 | 8,192 | n/a | 582.7 | n/a |
| 16 | 128 | 180.3 | 56.6 | 3.2 |
| 16 | 512 | 564.5 | 193.1 | 2.9 |
| 16 | 2,048 | 2,453.6 | 775.9 | 3.2 |
| 16 | 8,192 | n/a | 4,383.6 | n/a |

Nsight Compute on one launch each (`--launch-skip 3 --launch-count 1`, so clocks are locked and caches are cold, which makes the times longer than the loop above):

| rows x context | kernel | duration (us) | grid | SM throughput | DRAM throughput | warps active |
| --- | --- | --- | --- | --- | --- | --- |
| 1 x 2,048 | v1 | 1,310 | 32 | 3.8% | 0.5% | 6.6% |
| 1 x 2,048 | v2 | 383 | 32 | 15.5% | 1.6% | 20.6% |
| 16 x 512 | v1 | 1,220 | 512 | 16.1% | 2.0% | 45.2% |
| 16 x 512 | v2 | 383 | 512 | 57.3% | 6.4% | 84.6% |

Reading it. Both kernels are far from memory bound (DRAM at 6% or less), so the time is latency and occupancy, not bandwidth. v2 wins by about 3x because 8 lanes share each key (the key loads coalesce), all threads accumulate the values instead of `head_dim` of 256, and the shared memory no longer grows with the context. With one row there are only 32 blocks (one per head) on a GPU with far more SMs, which is why 1 row at context 8,192 still takes 583 us; splitting the context across blocks would be the next step and needs a scratch buffer the current signature does not have. Two things that did not help: replacing the shared-memory tree reductions with warp shuffles made the kernel 15 to 20% slower, so it was dropped. In the decode profile at 32 prompt and 32 output tokens (`scripts/profile.sh nsys b1`, 224 kernels per step) attention is 485 us of 7.3 ms (6.5%) because the context is at most 64 and the GEMVs take 85%; v2 was wired into `PagedRunner` in the next branch (below).

To reproduce: `nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/paged_attention_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/pa_bench && /tmp/pa_bench`, and for one launch under Nsight Compute `ncu --kernel-name "regex:^paged_attention_decode_v2_kernel" --launch-skip 3 --launch-count 1 /tmp/pa_bench 1 2048` (rows and context as arguments, kernel `paged_attention_decode_kernel` for v1).

## Paged attention v2 in the forward pass (Branch 38)

`PagedRunner` calls `paged_attention_decode_v2` instead of v1 (`scripts/reproduce.sh --skip-tests` on commit 79c1f91, 2026-10-20, RTX 5060 Laptop GPU). The context is no longer capped by the shared memory of v1 and the BF16 score rounding is unchanged, so the greedy serve tests, the CUDA Graph test and the paged HF comparison pass as before; the sampled forced-preemption test now allows 2 of 16 requests to differ, because a recomputed request runs in a batch of another size and one sampled token near a BF16 ulp flipped (request 15, token 44 of 48).

| | before (e07c022) | after (79c1f91) |
| --- | --- | --- |
| attention per decode step (eager profile, context at most 64) | 883 us (11.5% of kernel time) | 272 us (3.8%) |
| batch-1 TPOT with graphs | 8.43 ms | 7.96 ms |
| batch-1 weights bandwidth with graphs | 246 GB/s (76%) | 260 GB/s (80%) |
| bench tokens/s at max batch 1 / 4 / 16 with graphs | 117.9 / 426.5 / 1074.8 | 124.4 / 468.4 / 1386.2 |
| bench tokens/s at max batch 1 / 4 / 16 eager | 107.6 / 396.0 / 1011.9 | 114.3 / 426.4 / 1282.8 |

Reading it. Batch 16 gains the most (29% with graphs, 27% eager) and batch 1 gains 5% to 6%, which matches the attention time saved at batch 1 (0.6 ms of 8.4 ms); the larger gain at batch 16 is consistent with attention work growing with the number of rows while the weight GEMVs do not, but I did not profile batch 16 to confirm where its time went. Part of the difference between the two columns can be clock noise (the HuggingFace rows of the same run were slower than in the previous one, so the machine was not faster), so the direction and the attention row are solid and the third digit is not. The next step for attention is the one named above, splitting the context across blocks.

## Tuning add_rmsnorm (Branch 39)

The first version of `fast::add_rmsnorm` was latency bound at about 11 us for 1 to 64 rows: eight scalar 2-byte loads per thread, a second pass that read the sums back from global memory, two barriers and a shared `rstd`. The tuned kernel (used when the hidden size is a multiple of 8, at most 8192, and every pointer is 16-byte aligned) loads x and the residual with 128-bit loads, keeps the rounded sums in registers for the second pass, reduces with `__shfl_xor_sync` so every thread holds the total after one barrier, and writes with 128-bit stores. Other shapes keep the old scalar kernel. Nsight Compute, hidden 2048, release build, same method as the first table (`scripts/add_rmsnorm_bench.cu`):

| tokens | naive pair (us) | first fused (us) | tuned fused (us) | tuned / naive pair |
| --- | --- | --- | --- | --- |
| 1 | 11.8 (3.6 + 8.2) | 10.9 | 4.0 | 0.34 |
| 16 | 11.6 (3.9 + 7.6) | 10.7 | 4.3 | 0.37 |
| 64 | 14.2 (6.5 + 7.7) | 11.1 | 5.3 | 0.37 |

In the decode step of `scripts/reproduce.sh` (commit 1aa5e88) the two fused ops go from 92.4 + 84.7 = 177 us to 38.0 + 34.5 = 73 us per step, about 105 us of a 7.3 ms step (1.4%). The end-to-end effect is therefore small and sits inside the run-to-run spread: batch-1 TPOT with graphs 7.96 to 7.83 ms (260 to 264 GB/s, 80% to 82%), bench tokens/s with graphs at max batch 1 / 4 / 16 124.4 / 468.4 / 1386.2 to 126.7 / 473.0 / 1405.8. The same run shows attention at 326 us per step against 272 us before, a kernel this branch did not touch, which is the size of the noise. The kernel is now below the launch cost of the pair it replaces, so the fusion finally saves time as well as launches. The reduction order changed (8 consecutive elements per thread, warps combined in a fixed order), so the logits move by rounding as they did when the kernel was first wired in; the serve tests, which accept exact ties, pass unchanged.

## Split-context attention (Branch 40)

`paged_attention_decode_split` against `paged_attention_decode_v2`, RTX 5060 Laptop GPU, release build, 32 heads, 4 KV heads, head_dim 64, block size 16, every row at the full context. Time per call in us from a loop of 50 calls after 3 warm-up calls (`scripts/paged_attention_bench.cu`), the split column includes the merge kernel, the second run of two agreed within noise:

| rows | context | v2 | 1 split | 2 | 4 | 8 | 16 | best | v2 / best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 27.1 | 37.2 | 37.9 | 39.7 | 47.6 | 41.0 | v2 | 1.0 |
| 1 | 512 | 46.0 | 53.1 | 44.1 | 40.5 | 47.5 | 64.2 | 4 | 1.1 |
| 1 | 2,048 | 167.6 | 166.2 | 92.8 | 64.7 | 60.7 | 67.5 | 8 | 2.8 |
| 1 | 8,192 | 582.9 | 574.6 | 318.1 | 221.7 | 207.9 | 210.4 | 8 | 2.8 |
| 16 | 128 | 59.8 | 61.6 | 76.1 | 99.5 | 153.0 | 165.1 | v2 | 1.0 |
| 16 | 512 | 192.2 | 197.2 | 191.1 | 210.8 | 285.1 | 408.5 | 2 | 1.0 |
| 16 | 2,048 | 789.0 | 792.8 | 827.6 | 842.9 | 860.1 | 968.2 | v2 | 1.0 |
| 16 | 8,192 | 4,417.0 | 4,211.4 | 3,801.9 | 4,495.2 | 4,430.1 | 3,556.9 | 16 | 1.2 |

Nsight Compute on one launch each (clocks locked, cold caches, so longer than the loop), 1 row at 2,048 tokens: v2 383 us with DRAM throughput at 1.6%; split with 1 split 361.6 us for the partial kernel plus 5.0 us for the merge (1.7%); with 2 splits 185.5 + 4.7 us (3.3%). The runs I took for more splits were not labelled with their split count, so they are not recorded here.

Reading it. At batch 1 the grid of v2 is 32 blocks, fewer than the SMs can hold, and splitting the context into 4 to 8 pieces makes a 2,048-token context 2.8x faster and an 8,192-token context 2.8x faster, with 8 splits (256 blocks) the best or within 3% of the best from 2,048 tokens up. At 512 tokens the gain is 1.1x (4 splits) and at 128 tokens it loses, because a split holds too few tokens to pay for the second launch and the partial write. With 16 rows the grid is already 512 blocks, so splitting only adds work: it ties v2 at 2 splits, loses beyond that, and wins only at 8,192 tokens (1.2x at 16 splits, with a noisy 4-split point). One split costs 0 to 10 us over v2 (the extra merge launch), more at short contexts. A rule for the wiring branch: use `num_splits = 1` (or v2) when `batch * 32 >= 256` blocks already or the longest context is under 512 tokens, otherwise `num_splits` near `256 / (batch * 32)` rounded to a power of two, at most 8 to 16 and at least 256 tokens per split.

To reproduce: `nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/paged_attention_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/pa_bench && /tmp/pa_bench`.

## Split attention in the forward pass (Branch 42)

`PagedRunner` now calls `paged_attention_decode_split` when `attention_splits(rows, heads, longest_context)` asks for more than one split (1 to 4 rows and a context of 512 tokens or more; with TinyLlama's 2,048 positions one row gets 2 splits from 512 tokens and 4 from 1,024, since 8 would need 2,048). `engine bench`, RTX 5060 Laptop GPU, release build, main (b597d93) against this branch (e298faf), 64 output tokens, warmup 1, TPOT p50 in ms of two runs each (run 1 / run 2):

| max batch | prompt | splits | v2 eager | split eager | v2 graphs | this branch graphs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 1 | 8.90 / 8.86 | 8.84 / 10.08 | 8.32 / 8.53 | 8.44 / 8.43 |
| 1 | 512 | 2 | 9.63 / 9.62 | 9.28 / 9.51 | 9.19 / 9.38 | 9.23 / 9.79 |
| 1 | 1,024 | 4 | 10.55 / 10.73 | 9.34 / 9.68 | 9.94 / 10.09 | 10.05 / 10.02 |
| 1 | 1,984 | 4 | 14.68 / 14.70 | 10.19 / 10.94 | 13.97 / 14.08 | 14.05 / 14.03 |
| 4 | 1,984 | 2 when 4 rows decode | 50.93 / 51.19 | 50.71 / 51.06 | 50.10 / 50.43 | 50.23 / 50.48 |

Reading it. At batch 1 eager the split takes the long-context TPOT from 14.7 to 10.2 to 10.9 ms (1.4x). That saving of 3.8 to 4.5 ms per token is more than the 22 x (167.6 - 64.7) us = 2.3 ms the warm-cache loop above predicts for 4 splits at 2,048 tokens and less than the cold-cache Nsight Compute numbers (v2 383 us per launch) would give; inside a step attention runs right after GEMMs that stream 2 GB of weights through L2, so its cache state is likely between the two. Not checked with a profile yet; at 1,024 tokens the gain is 1.1x and at 512 it is inside the noise. Short contexts take the v2 path and are unchanged (the 10.08 ms eager run at 64 tokens is a host hiccup; the graph runs of the same build agree with main). The graph columns do not move: `DecodeGraphs` captures each bucket on a dummy batch with no context, so the captured step always runs v2 and a replay cannot change its grid, which makes eager faster than graphs from about 1,024 tokens on. With 4 requests at 1,984 tokens the TPOT is dominated by the prefills that stall the decodes (each step is all prefill or all decode), so the attention gain does not show. The short-prompt workload of `scripts/reproduce.sh` (64 prompt and 64 output tokens) never splits, so `docs/results.md` keeps its numbers.

To reproduce one cell: `build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --out /tmp/split.csv` (add `--graphs` for the graph column).

## Split attention in CUDA Graphs (Branch 43)

`DecodeGraphs` now captures one graph per (bucket, split count): for each bucket it walks the contexts from 0 to the runner's max context and captures the step on a dummy batch at the first context that gives a new `attention_splits(bucket, heads, context)`, and `forward` replays the graph of the padded batch's split count. With TinyLlama at a 2,048-token max context that is 11 graphs instead of 5 (bucket 1: 1, 2, 4, 8 splits; bucket 2: 1, 2, 4; bucket 4: 1, 2; buckets 8 and 16: 1). `engine bench`, RTX 5060 Laptop GPU, release build of this branch, same settings as the table above (64 output tokens, warmup 1), TPOT p50 in ms of two runs each, with the graph column of main copied from that table:

| max batch | prompt | splits | eager | main graphs | this branch graphs |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 1 | 8.85 / 9.15 | 8.44 / 8.43 | 8.04 / 8.34 |
| 1 | 512 | 2 | 9.47 / 9.15 | 9.23 / 9.79 | 9.04 / 9.09 |
| 1 | 1,024 | 4 | 9.41 / 9.72 | 10.05 / 10.02 | 8.26 / 8.33 |
| 1 | 1,984 | 4 | 9.93 / 10.60 | 14.05 / 14.03 | 9.13 / 9.12 |
| 4 | 1,984 | 2 when 4 rows decode | 50.45 / 50.41 | 50.23 / 50.48 | 49.38 / 49.49 |

Reading it. Graphs are faster than eager again at every length: at a 1,984-token prompt the replayed step goes from 14.0 to 9.1 ms (1.5x) and sits about 1 ms under eager, the same launch saving graphs give at 64 tokens. At 64 tokens the split count is 1, so the replay is the old graph and the difference to main is clock noise. Batch 4 at 1,984 tokens is still dominated by prefills that stall the decodes (each step is all prefill or all decode). The extra graphs cost capture time once at startup and some device memory for the executable graphs; neither was measured. Compute-sanitizer memcheck on the 12 `DecodeGraphsTest` tests: 0 errors, 0 bytes leaked. Reproduce one cell with the command above plus `--graphs`.

## write_kv (Branch 64)

Measured with Nsight Compute on the RTX 5060 Laptop GPU on 2026-10-29 (release build, commit 619676d, default clock control), in the real forward pass: `ncu --profile-from-start off --kernel-name regex:write_kv_kernel --metrics gpu__time_duration.sum,dram__bytes.sum,launch__grid_size` around `engine bench --max-batch N`, one launch per layer (22 per step), medians over the layers. `write_kv` has a single version, one thread per element with a 256-thread block, so there is no before and after to compare: this is the only measurement.

| rows in the step | grid blocks | launches | median time (us) | range (us) | DRAM (KB) | DRAM throughput |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (decode, batch 1) | 1 | 22 | 3.87 | 3.81 to 4.16 | 7.9 | 0.6% of peak |
| 16 (decode, batch 16) | 16 | 110 | 3.78 | 3.62 to 4.16 | 23.3 | not recorded |
| 64 (prefill, one prompt) | 64 | 22 | 4.32 | 4.10 to 6.18 | 73.2 | 4.7% of peak |
| 512 (prefill, 16 prompts) | 512 | 22 | 9.34 | 9.06 to 11.20 | 533.8 | 18% of the measured 323.6 GB/s |

Reading it. A decode step copies the K and V slices of one row, 2 KB in and 2 KB out, so the kernel is nowhere near a bandwidth limit: its time is the floor of a launch plus one dependent chain of loads (the slot, then the data) and stores, about 3.8 us at 1 and 16 rows. Nsight Compute flushes the caches and locks the clocks to base, so it overstates the time in a running engine: the Nsight Systems profile of the decode step (`docs/results.md`) measures 1.5 us per launch, 32 us and 0.4% of a batch-1 step, where 22 launches at 3.8 us would be about 85 us and 1.1%. Either way it is noise next to the 7.8 ms step. It only grows with work at prefill sizes, 9.3 us for 512 rows, which is 18% of the DRAM roof and a one-off cost per prompt. There is no kernel to optimize here; the measurement closes the gap in the "explained" goal.


## Sampling (Branch 65)

`greedy` and `sample` against a full sort of the vocabulary, RTX 5060 Laptop GPU, release build on commit b10575c, 2026-10-31, vocab 32,000, temperature 0.7, logits drawn from a normal distribution with a standard deviation of 2.5 (synthetic, so how many tokens top-p keeps is not the model's). Time per call in us from a loop of 50 calls after 3 warm-up calls (`scripts/sample_bench.cu`), two runs (run 1 / run 2). The full sort column is the cost of sorting every row with `cub::DeviceSegmentedRadixSort` after scaling it to float, before any prefix sum or draw, so it is a lower bound for a sort-based top-p:

| rows | greedy | temperature | top-k 50 | top-p 0.9 | top-k 50 + top-p 0.9 | full sort alone |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.9 / 22.4 | 425.2 / 425.2 | 126.8 / 111.6 | 325.2 / 325.2 | 172.6 / 172.8 | 271.2 / 282.0 |
| 16 | 20.4 / 19.2 | 425.7 / 425.7 | 119.7 / 119.2 | 346.7 / 346.4 | 174.6 / 175.3 | 235.1 / 238.9 |

Nsight Compute on one temperature-only launch at 1 row (`--launch-skip 3 --launch-count 1`): 703 us, grid 1 block of 1,024 threads, 48 registers per thread, SM throughput 3.3% and DRAM 0.05% of the GPU, and on the one busy SM the FP64 pipe is active for 1,008,000 of its 1,193,622 cycles (84%) while FP64 instructions are 63,000 of 478,550 (13%).

The same launch as a table, and the throwaway variants of `gumbel()` timed in the same session as the unchanged kernel (the bench loop, interleaved runs; not committed code):

| metric | value |
| --- | --- |
| duration (us) | 703.0 |
| FP64 pipe active (cycles) | 1,008,000 |
| SM cycles elapsed | 1,193,622 |
| FP64 pipe busy | 84% |
| FP64 instructions | 63,000 |
| instructions | 478,550 |
| FP64 share of instructions | 13% |

| variant of `gumbel()` | 1 row, temperature (us) | 16 rows, temperature (us) | 1 row, top-k 50 (us) |
| --- | --- | --- | --- |
| as written (double `log`) | 467.9 / 471.5 | 474.9 / 475.2 | 115.2 / 131.4 |
| float `u` and `__logf` | 95.5 / 87.5 / 64.5 | 91.1 / 71.2 / 85.7 | 181.1 / 191.3 / 183.7 |
| logs removed | 54.4 | 77.8 | 174.4 |

Reading it. The threshold searches do what they were written for: top-k 50 finds its cut with 32 passes of a count over the row and is about 2x faster than sorting the row alone, before the sort-based version has done any of its remaining work. The surprise is the plain temperature column. Every one of the 32,000 tokens gets Gumbel noise, and `gumbel()` computes `-log(-log(u))` in double precision; on this GPU FP64 runs at a small fraction of the FP32 rate, so the FP64 pipe is busy for 84% of the kernel's cycles and the draw costs 425 us, more than with top-k, where the noise is only computed for the 50 tokens above the cut. Top-p 0.9 keeps a large part of these synthetic logits, so it pays for most of that noise as well as for its own search of the mass threshold (an `expf` over the row per pass). Rows do not change the times because each row is one block: 16 rows run on 16 SMs side by side, and 1 row leaves 25 of the 26 SMs idle. One sampled draw per decode step at 425 us would be about 5% of the 7.8 ms batch-1 step; the end-to-end effect was not measured here. Not changed in this branch, because it changes the sampled tokens of a fixed seed: in a throwaway copy of `sample.cu` with `u` in float and `__logf`, the temperature column fell to 65 to 95 us at both row counts (four interleaved runs against 468 to 475 us for the unchanged kernel in the same session), and with the logs removed entirely to 54 to 78 us. The same copy made the top-k column slower (180 to 315 us against 115 to 140 us), which I have not explained, so it would need its own profile before it is proposed.

To reproduce: `nvcc -std=c++20 -O3 -arch=sm_120 -Iinclude -Isrc scripts/sample_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/sample_bench && /tmp/sample_bench` (arguments: rows, then vocab), and for one launch `ncu --kernel-name regex:sample_kernel --launch-skip 3 --launch-count 1 --metrics sm__pipe_fp64_cycles_active.max,sm__cycles_elapsed.max /tmp/sample_bench 1`.

## Split attention occupancy (Branch 66)

Why splitting helps at batch 1, measured with Nsight Compute on the RTX 5060 Laptop GPU on 2026-10-31 (release build, commit 19e3d13, `scripts/paged_attention_bench.cu` at 1 row and 2,048 tokens, one launch per kernel after 3 warm-up launches, clocks locked and caches cold, so the times are longer than the loop of "Split-context attention"). Both kernels launch 256-thread blocks at 40 registers per thread; registers and warps each allow 6 blocks per SM (shared memory would allow 14), so the 26 SMs hold 156 blocks in one wave.

| kernel | splits | grid blocks | waves per SM | warps active | time (us) | DRAM throughput |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 1 | 32 | 0.21 | 20.5% | 382.5 | 1.6% |
| split partial | 1 | 32 | 0.21 | 20.5% | 361.3 | 1.7% |
| split partial | 2 | 64 | 0.41 | 40.9% | 186.0 | 3.3% |
| split partial | 4 | 128 | 0.82 | 78.8% | 106.7 | 5.7% |
| split partial | 8 | 256 | 1.64 | 80.1% | 102.2 | 6.0% |

The merge kernel after 8 splits is 32 blocks and 6.1 us.

Reading it. With one row the grid is one block per query head, 32 blocks for 156 slots, so a fifth of the GPU's warps are ever resident and each block walks the whole context alone. Every doubling of the split count doubles the grid and halves the time while the grid still fits in one wave (up to 4 splits, 128 blocks). At 8 splits the grid is 1.64 waves: the first 156 blocks run together and the last 100 run in a second, partly empty wave, so the time barely moves (106.7 to 102.2 us) although each block has half the work. DRAM stays at 6% or less throughout, so the kernel is bound by latency and the number of blocks in flight, not by bandwidth. This is the measurement behind the rule in "Split attention in the forward pass": aim for about 256 blocks, since the gain stops near one full wave.

To reproduce: build `/tmp/pa_bench` as in "Split-context attention", then `ncu --kernel-name regex:paged_attention_split_partial_kernel --launch-skip 162 --launch-count 1 --metrics launch__grid_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_warps,gpu__time_duration.sum /tmp/pa_bench 1 2048` for 8 splits (the bench runs 53 calls per split count in the order 1, 2, 4, 8, 16, so skip 3, 56 and 109 for 1, 2 and 4 splits), and `--kernel-name regex:^paged_attention_decode_v2_kernel --launch-skip 3` for v2.

## Attention shared memory (Branch 67)

Dynamic shared memory per thread block of the two decode attention kernels, computed from their launch code (`src/kernels/paged_attention.cu` and `src/kernels/paged_attention_v2.cu`, commit 19e3d13), 256 threads per block and head_dim 64. v1 reserves one float per token of the block table's capacity plus one per thread and the query in BF16: `(capacity + 256) * 4 + 64 * 2` bytes. v2 reserves three floats per thread, the 64-float accumulator and the query: `3 * 256 * 4 + 64 * 4 + 64 * 2` bytes, whatever the context. CUDA's default limit for dynamic shared memory is 48 KB (49,152 bytes) per block without an opt-in.

| context | v1 bytes | v2 bytes |
| --- | --- | --- |
| 128 | 1,664 | 3,456 |
| 512 | 3,200 | 3,456 |
| 2,048 | 9,344 | 3,456 |
| 8,192 | n/a | 3,456 |

Reading it. v1's context limit is not shared memory running out: at its cap of 2,048 tokens it uses 9,344 bytes, 19% of the default limit, and the formula would reach the limit only near 12,000 tokens. The cap is `kMaxContext = 2048` in the host code, which rejects any block table longer than that (TinyLlama's full context), and that check is why the 8,192-token rows of "Paged attention v2" read n/a for v1 (it would have needed 33,920 bytes). The sentence there that v1 "cannot run a context of 8,192 because its scores live in shared memory" is right about the cause of the design and not about the limit. Neither kernel is limited by shared memory per SM at these sizes ("Split attention occupancy" measured v2 at 6 blocks per SM from registers and warps, 14 from shared memory). The bytes were not yet confirmed with Nsight Compute's `launch__shared_mem_per_block_dynamic`, because the GPU was unavailable when this was written.
