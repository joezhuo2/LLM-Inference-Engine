Engine runs measured 2026-11-05 at commit 1693c46 on NVIDIA GeForce RTX 5060 Laptop GPU; vLLM runs measured 2026-11-05 (vllm 0.32.0, torch 2.13.0).

SLO: TTFT at most 500 ms and TPOT at most 50 ms; goodput is the requests per second that meet both.

| system | QPS | output tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | SLO met | goodput req/s | preemptions | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| engine + graphs | 1 | 168 | 46 | 102 | 8.4 | 11.2 | 100% | 1.14 | 0 | 3 |
| engine + graphs | 2 | 330 | 46 | 115 | 9.4 | 12.9 | 100% | 2.24 | 0 | 3 |
| engine + graphs | 3 | 486 | 48 | 123 | 10.7 | 15.6 | 100% | 3.30 | 0 | 3 |
| engine + graphs | 4 | 635 | 64 | 605 | 12.3 | 17.8 | 98% | 4.21 | 0 | 3 |
| engine + graphs | 5 | 765 | 80 | 1192 | 14.1 | 17.2 | 85% | 4.42 | 0 | 3 |
| engine + graphs | 6 | 839 | 240 | 1991 | 15.3 | 19.7 | 64% | 3.66 | 0 | 3 |
| engine + graphs | 8 | 885 | 1415 | 5007 | 15.5 | 19.9 | 25% | 1.50 | 0 | 3 |
| engine + graphs, budget 256 | 1 | 168 | 52 | 121 | 8.3 | 10.3 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 256 | 2 | 330 | 56 | 136 | 9.2 | 12.1 | 100% | 2.24 | 0 | 3 |
| engine + graphs, budget 256 | 3 | 486 | 57 | 146 | 10.2 | 14.0 | 100% | 3.30 | 0 | 3 |
| engine + graphs, budget 256 | 4 | 635 | 70 | 413 | 11.5 | 16.2 | 99% | 4.28 | 0 | 3 |
| engine + graphs, budget 256 | 5 | 776 | 80 | 1033 | 13.2 | 16.4 | 87% | 4.57 | 0 | 3 |
| engine + graphs, budget 256 | 6 | 862 | 116 | 1642 | 14.2 | 17.5 | 69% | 4.05 | 0 | 3 |
| engine + graphs, budget 256 | 8 | 919 | 1117 | 4316 | 14.6 | 18.6 | 26% | 1.61 | 0 | 3 |
| engine + graphs, budget 512 | 1 | 168 | 46 | 107 | 8.3 | 10.8 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 512 | 2 | 330 | 48 | 117 | 9.2 | 12.3 | 100% | 2.24 | 0 | 3 |
| engine + graphs, budget 512 | 3 | 486 | 52 | 138 | 10.3 | 14.6 | 100% | 3.30 | 0 | 3 |
| engine + graphs, budget 512 | 4 | 635 | 67 | 477 | 11.7 | 16.9 | 99% | 4.28 | 0 | 3 |
| engine + graphs, budget 512 | 5 | 772 | 74 | 1087 | 13.6 | 17.2 | 86% | 4.50 | 0 | 3 |
| engine + graphs, budget 512 | 6 | 854 | 119 | 1755 | 14.5 | 18.8 | 68% | 3.97 | 0 | 3 |
| engine + graphs, budget 512 | 8 | 912 | 1147 | 4466 | 14.8 | 19.2 | 26% | 1.60 | 0 | 3 |
| engine + graphs, budget 2048 | 1 | 168 | 46 | 106 | 8.4 | 10.8 | 100% | 1.14 | 0 | 3 |
| engine + graphs, budget 2048 | 2 | 330 | 48 | 120 | 9.2 | 12.3 | 100% | 2.24 | 0 | 3 |
| engine + graphs, budget 2048 | 3 | 486 | 51 | 130 | 10.4 | 14.6 | 100% | 3.30 | 0 | 3 |
| engine + graphs, budget 2048 | 4 | 635 | 65 | 506 | 11.8 | 17.3 | 98% | 4.25 | 0 | 3 |
| engine + graphs, budget 2048 | 5 | 769 | 80 | 1105 | 13.6 | 17.2 | 85% | 4.44 | 0 | 3 |
| engine + graphs, budget 2048 | 6 | 853 | 139 | 1761 | 14.5 | 18.7 | 68% | 3.96 | 0 | 3 |
| engine + graphs, budget 2048 | 8 | 916 | 1153 | 4397 | 14.9 | 18.9 | 26% | 1.61 | 0 | 3 |
| vLLM | 1 | 168 | 42 | 68 | 7.6 | 8.5 | 100% | 1.14 | 0 | 3 |
| vLLM | 2 | 331 | 28 | 48 | 7.5 | 8.1 | 100% | 2.25 | 0 | 3 |
| vLLM | 3 | 489 | 29 | 39 | 7.6 | 8.3 | 100% | 3.33 | 0 | 3 |
| vLLM | 4 | 641 | 30 | 41 | 8.1 | 8.4 | 100% | 4.36 | 0 | 3 |
| vLLM | 5 | 789 | 31 | 55 | 8.2 | 8.7 | 100% | 5.36 | 0 | 3 |
| vLLM | 6 | 933 | 31 | 90 | 8.3 | 8.7 | 100% | 6.34 | 0 | 3 |
| vLLM | 8 | 1208 | 34 | 507 | 8.5 | 8.8 | 98% | 8.07 | 0 | 3 |

Gaps between consecutive output tokens of one request, from the engine's step traces: every token of every request, medians of the runs' p50, p99 and largest gap.

| system | QPS | gap p50 ms | gap p99 ms | largest gap ms | runs |
|---|---|---|---|---|---|
| engine + graphs | 1 | 8.0 | 9.6 | 158 | 3 |
| engine + graphs | 2 | 8.5 | 51.1 | 161 | 3 |
| engine + graphs | 3 | 9.1 | 75.8 | 178 | 3 |
| engine + graphs | 4 | 9.9 | 86.8 | 201 | 3 |
| engine + graphs | 5 | 11.0 | 96.5 | 212 | 3 |
| engine + graphs | 6 | 11.4 | 99.3 | 182 | 3 |
| engine + graphs | 8 | 11.5 | 101.7 | 173 | 3 |
| engine + graphs, budget 256 | 1 | 8.0 | 26.4 | 65 | 3 |
| engine + graphs, budget 256 | 2 | 8.5 | 36.4 | 65 | 3 |
| engine + graphs, budget 256 | 3 | 9.1 | 39.8 | 66 | 3 |
| engine + graphs, budget 256 | 4 | 9.8 | 44.9 | 67 | 3 |
| engine + graphs, budget 256 | 5 | 10.8 | 49.9 | 68 | 3 |
| engine + graphs, budget 256 | 6 | 11.3 | 54.4 | 70 | 3 |
| engine + graphs, budget 256 | 8 | 11.5 | 59.0 | 71 | 3 |
| engine + graphs, budget 512 | 1 | 8.0 | 10.8 | 97 | 3 |
| engine + graphs, budget 512 | 2 | 8.5 | 44.7 | 98 | 3 |
| engine + graphs, budget 512 | 3 | 9.1 | 67.8 | 98 | 3 |
| engine + graphs, budget 512 | 4 | 9.8 | 76.8 | 100 | 3 |
| engine + graphs, budget 512 | 5 | 10.9 | 81.8 | 103 | 3 |
| engine + graphs, budget 512 | 6 | 11.4 | 84.8 | 105 | 3 |
| engine + graphs, budget 512 | 8 | 11.5 | 86.9 | 106 | 3 |
| engine + graphs, budget 2048 | 1 | 8.1 | 10.9 | 96 | 3 |
| engine + graphs, budget 2048 | 2 | 8.5 | 44.9 | 100 | 3 |
| engine + graphs, budget 2048 | 3 | 9.1 | 68.8 | 102 | 3 |
| engine + graphs, budget 2048 | 4 | 9.9 | 78.4 | 116 | 3 |
| engine + graphs, budget 2048 | 5 | 10.9 | 85.6 | 163 | 3 |
| engine + graphs, budget 2048 | 6 | 11.4 | 91.1 | 158 | 3 |
| engine + graphs, budget 2048 | 8 | 11.5 | 92.5 | 160 | 3 |
