Engine runs measured 2026-11-11 at commit 1feaba9 on NVIDIA GeForce RTX 5060 Laptop GPU; vLLM runs measured 2026-11-11 (vllm 0.32.0, torch 2.13.0).

SLO: TTFT at most 500 ms and TPOT at most 50 ms; goodput is the requests per second that meet both.

| system | QPS | output tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | SLO met | goodput req/s | preemptions | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| engine + graphs | 1 | 167 | 35 | 103 | 9.5 | 11.4 | 100% | 1.14 | 0 | 3 |
| engine + graphs | 2 | 329 | 34 | 67 | 10.1 | 12.6 | 100% | 2.23 | 0 | 3 |
| engine + graphs | 3 | 480 | 35 | 67 | 11.5 | 14.7 | 100% | 3.26 | 0 | 3 |
| engine + graphs | 4 | 632 | 38 | 467 | 12.6 | 16.5 | 99% | 4.26 | 0 | 3 |
| engine + graphs | 5 | 763 | 45 | 1094 | 14.4 | 16.8 | 86% | 4.45 | 0 | 3 |
| engine + graphs | 6 | 845 | 93 | 1669 | 15.3 | 17.4 | 67% | 3.83 | 0 | 3 |
| engine + graphs | 8 | 898 | 1304 | 4592 | 15.4 | 17.6 | 25% | 1.53 | 0 | 3 |
| engine + graphs, budget 256 | 1 | 167 | 43 | 94 | 9.3 | 10.9 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 256 | 2 | 327 | 42 | 85 | 9.8 | 12.1 | 100% | 2.23 | 0 | 3 |
| engine + graphs, budget 256 | 3 | 484 | 45 | 99 | 10.7 | 13.7 | 100% | 3.29 | 0 | 3 |
| engine + graphs, budget 256 | 4 | 626 | 48 | 289 | 12.0 | 15.2 | 100% | 4.25 | 0 | 3 |
| engine + graphs, budget 256 | 5 | 771 | 55 | 921 | 13.5 | 15.2 | 87% | 4.54 | 0 | 3 |
| engine + graphs, budget 256 | 6 | 858 | 70 | 1447 | 14.3 | 15.9 | 74% | 4.32 | 0 | 3 |
| engine + graphs, budget 256 | 8 | 924 | 1077 | 4062 | 14.8 | 16.7 | 26% | 1.62 | 0 | 3 |
| engine + graphs, budget 512 | 1 | 167 | 36 | 101 | 9.3 | 10.9 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 512 | 2 | 329 | 36 | 69 | 10.0 | 12.2 | 100% | 2.23 | 0 | 3 |
| engine + graphs, budget 512 | 3 | 481 | 38 | 82 | 11.1 | 14.1 | 100% | 3.27 | 0 | 3 |
| engine + graphs, budget 512 | 4 | 632 | 41 | 295 | 11.9 | 15.4 | 100% | 4.30 | 0 | 3 |
| engine + graphs, budget 512 | 5 | 771 | 46 | 941 | 13.6 | 15.4 | 87% | 4.54 | 0 | 3 |
| engine + graphs, budget 512 | 6 | 855 | 58 | 1463 | 14.5 | 16.1 | 74% | 4.31 | 0 | 3 |
| engine + graphs, budget 512 | 8 | 920 | 1098 | 4143 | 14.9 | 17.0 | 26% | 1.62 | 0 | 3 |
| engine + graphs, budget 2048 | 1 | 167 | 35 | 89 | 9.3 | 11.3 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 2048 | 2 | 329 | 36 | 69 | 9.8 | 12.2 | 100% | 2.24 | 0 | 3 |
| engine + graphs, budget 2048 | 3 | 484 | 38 | 75 | 10.9 | 14.1 | 100% | 3.29 | 0 | 3 |
| engine + graphs, budget 2048 | 4 | 626 | 42 | 298 | 12.2 | 15.5 | 100% | 4.25 | 0 | 3 |
| engine + graphs, budget 2048 | 5 | 771 | 46 | 947 | 13.6 | 15.4 | 87% | 4.54 | 0 | 3 |
| engine + graphs, budget 2048 | 6 | 856 | 58 | 1462 | 14.4 | 16.1 | 74% | 4.32 | 0 | 3 |
| engine + graphs, budget 2048 | 8 | 917 | 1099 | 4211 | 14.9 | 17.0 | 26% | 1.61 | 0 | 3 |
| vLLM | 1 | 167 | 60 | 121 | 8.8 | 11.5 | 100% | 1.14 | 0 | 3 |
| vLLM | 2 | 330 | 32 | 53 | 8.5 | 9.1 | 100% | 2.24 | 0 | 3 |
| vLLM | 3 | 482 | 45 | 122 | 9.6 | 14.3 | 100% | 3.28 | 0 | 3 |
| vLLM | 4 | 638 | 36 | 72 | 8.9 | 12.1 | 100% | 4.34 | 0 | 3 |
| vLLM | 5 | 785 | 34 | 63 | 8.8 | 9.5 | 100% | 5.34 | 0 | 3 |
| vLLM | 6 | 927 | 36 | 313 | 9.0 | 10.6 | 100% | 6.30 | 0 | 3 |
| vLLM | 8 | 1192 | 38 | 644 | 9.2 | 9.5 | 92% | 7.49 | 0 | 3 |

Gaps between consecutive output tokens of one request, from the engine's step traces: every token of every request, medians of the runs' p50, p99 and largest gap.

| system | QPS | gap p50 ms | gap p99 ms | largest gap ms | runs |
|---|---|---|---|---|---|
| engine + graphs | 1 | 9.1 | 18.7 | 110 | 3 |
| engine + graphs | 2 | 9.5 | 39.4 | 93 | 3 |
| engine + graphs | 3 | 10.2 | 50.1 | 101 | 3 |
| engine + graphs | 4 | 11.0 | 54.9 | 108 | 3 |
| engine + graphs | 5 | 12.4 | 55.8 | 113 | 3 |
| engine + graphs | 6 | 12.8 | 57.9 | 120 | 3 |
| engine + graphs | 8 | 12.9 | 59.6 | 86 | 3 |
| engine + graphs, budget 256 | 1 | 9.0 | 21.4 | 41 | 3 |
| engine + graphs, budget 256 | 2 | 9.4 | 26.0 | 42 | 3 |
| engine + graphs, budget 256 | 3 | 10.1 | 29.1 | 41 | 3 |
| engine + graphs, budget 256 | 4 | 11.1 | 30.7 | 41 | 3 |
| engine + graphs, budget 256 | 5 | 12.1 | 31.2 | 41 | 3 |
| engine + graphs, budget 256 | 6 | 12.7 | 32.1 | 47 | 3 |
| engine + graphs, budget 256 | 8 | 13.0 | 32.7 | 42 | 3 |
| engine + graphs, budget 512 | 1 | 9.1 | 13.8 | 67 | 3 |
| engine + graphs, budget 512 | 2 | 9.5 | 34.2 | 53 | 3 |
| engine + graphs, budget 512 | 3 | 10.1 | 41.9 | 54 | 3 |
| engine + graphs, budget 512 | 4 | 10.9 | 45.8 | 56 | 3 |
| engine + graphs, budget 512 | 5 | 12.1 | 47.9 | 63 | 3 |
| engine + graphs, budget 512 | 6 | 12.7 | 48.5 | 59 | 3 |
| engine + graphs, budget 512 | 8 | 13.0 | 50.9 | 58 | 3 |
| engine + graphs, budget 2048 | 1 | 9.0 | 13.1 | 54 | 3 |
| engine + graphs, budget 2048 | 2 | 9.3 | 34.1 | 66 | 3 |
| engine + graphs, budget 2048 | 3 | 10.1 | 41.5 | 68 | 3 |
| engine + graphs, budget 2048 | 4 | 11.1 | 46.1 | 65 | 3 |
| engine + graphs, budget 2048 | 5 | 12.1 | 48.3 | 72 | 3 |
| engine + graphs, budget 2048 | 6 | 12.7 | 48.9 | 73 | 3 |
| engine + graphs, budget 2048 | 8 | 13.0 | 51.4 | 77 | 3 |
