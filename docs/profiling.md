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
