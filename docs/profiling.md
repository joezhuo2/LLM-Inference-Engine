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
