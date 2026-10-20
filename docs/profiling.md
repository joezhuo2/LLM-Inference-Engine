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

Reading it. The kernels are the same and take the same time (busy 7.9 ms against 7.8 ms), so the whole gain is the host side: the 267 launches of an eager step leave the GPU idle for 11% of the step, and one `cudaGraphLaunch` removes it. The gain shrinks with the batch because a larger batch makes the kernels longer, so the CPU launches stay ahead of the GPU and the gaps were already smaller. Prefill steps are not graphed and keep their gaps (about 9% idle).

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

Reading it. Both kernels are far from memory bound (DRAM at 6% or less), so the time is latency and occupancy, not bandwidth. v2 wins by about 3x because 8 lanes share each key (the key loads coalesce), all threads accumulate the values instead of `head_dim` of 256, and the shared memory no longer grows with the context. With one row there are only 32 blocks (one per head) on a GPU with far more SMs, which is why 1 row at context 8,192 still takes 583 us; splitting the context across blocks would be the next step and needs a scratch buffer the current signature does not have. Two things that did not help: replacing the shared-memory tree reductions with warp shuffles made the kernel 15 to 20% slower, so it was dropped. In the decode profile at 32 prompt and 32 output tokens (`scripts/profile.sh nsys b1`, 224 kernels per step) attention is 485 us of 7.3 ms (6.5%) because the context is at most 64 and the GEMVs take 85%; v2 is not wired into `PagedRunner`, so that profile is unchanged by this branch.

To reproduce: `nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/paged_attention_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/pa_bench && /tmp/pa_bench`, and for one launch under Nsight Compute `ncu --kernel-name "regex:^paged_attention_decode_v2_kernel" --launch-skip 3 --launch-count 1 /tmp/pa_bench 1 2048` (rows and context as arguments, kernel `paged_attention_decode_kernel` for v1).
