# Exercise: sum reduction of 16M floats

Three ways to sum 16,777,216 floats (64 MiB) on an RTX 5060 Laptop GPU: naive atomics, shared memory, and warp shuffle. Each result was checked to match the expected sum of 1.67772e+07 (2^24, which fits exactly in a float).

## Setup

| Item | Value |
| --- | --- |
| GPU | RTX 5060 Laptop (sm_120), Balanced power mode |
| Build | release preset, CUDA 13.3 |
| Input | 16,777,216 floats, all 1.0 |
| Launch config | 65,536 blocks of 256 threads, one element per thread |
| Timing | cudaEvent around one launch, warmup launch first, median of 5 launches per run, 3 runs of the executable |
| Reference peak | 323.6 GB/s from `bandwidth_test` |

## Results

| Method | Run 1 median (ms) | Run 2 median (ms) | Run 3 median (ms) | Reported (ms) | Speedup vs atomics | Effective bandwidth (GB/s) | % of peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive atomics | 20.714 | 20.713 | 20.714 | 20.714 | 1.0x | 3.2 | 1.0% |
| Shared memory | 0.501 | 0.499 | 0.500 | 0.500 | 41.4x | 134.3 | 41.5% |
| Warp shuffle | 0.347 | 0.346 | 0.347 | 0.347 | 59.8x | 193.7 | 59.8% |

The reported time is the median of the three run medians. Effective bandwidth is 67,108,864 bytes divided by that time, counting only the input read. The warp shuffle version is about 1.44x faster than shared memory and reaches about 60 percent of the peak measured by `bandwidth_test`.

## Why the three differ

The naive version has every thread issue an atomic add to a single global address, so the 16M updates serialize on one memory location. That is why it runs at about 3 GB/s, around 1 percent of peak.

The shared memory version reduces within each block first using a tree reduction, so only one atomic per block reaches global memory. The cost moves to shared memory traffic and `__syncthreads()` barriers between the reduction steps.

The warp shuffle version reduces within each warp using register-to-register shuffles, which need no shared memory and no barriers at that level, and it only combines warp results across the block at the end. Fewer barriers and fewer shared memory accesses give it the lead.

## Notes and caveats

- Timings are stable: the three run medians agree within 0.5 percent for every method, and the fastest and slowest of the individual launches stay within about 2.5 percent of each other. An earlier version without a warmup launch gave noisier and slower numbers (for example 22.1 ms for atomics), so the warmup removed first-launch overhead.
- All-ones input makes the sum exact, so it checks for dropped elements but not for floating point ordering effects. Correctness on random data is checked in the gtest file with a relative tolerance.
- The best kernel reaches 60 percent of measured peak. Each thread handles a single element, so there is little work per thread and no vectorized loads. Having each thread loop over several elements (a grid-stride loop) or read `float4` would likely close part of the gap, and the `ncu` profile should confirm where the remaining time goes.