Measured 2026-11-16T20:21:01Z at commit eaddcb7: one request per run, median of 3 runs.

TTFT in ms against prompt tokens, median (min to max):

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine, paged prefill | 16.3 (15.8 to 16.5) | 100.6 (100.5 to 101.7) | 348.4 (344.5 to 356.4) | 1186.6 (1181.1 to 1189.4) |
| engine | 11.0 (10.8 to 11.0) | 38.0 (37.9 to 38.3) | 75.9 (75.5 to 76.9) | 164.3 (162.8 to 165.1) |
| engine + graphs | 11.7 (10.8 to 12.5) | 38.1 (37.9 to 38.3) | 75.9 (75.8 to 76.0) | 164.7 (162.6 to 165.2) |
| engine + graphs, budget 256 | 11.2 (10.9 to 16.3) | 41.6 (41.1 to 41.7) | 87.7 (86.9 to 87.9) | 195.3 (192.9 to 195.7) |
| llama.cpp -ub 512 | 16.7 (16.0 to 19.6) | 46.5 (45.7 to 48.9) | 103.4 (101.8 to 111.8) | 186.0 (185.5 to 195.3) |
| llama.cpp -ub 2048 | 16.6 (16.4 to 17.9) | 45.7 (44.1 to 47.1) | 89.1 (88.5 to 92.4) | 195.7 (193.6 to 198.4) |

Median TTFT in ms per prompt token:

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine, paged prefill | 0.127 | 0.197 | 0.340 | 0.579 |
| engine | 0.086 | 0.074 | 0.074 | 0.080 |
| engine + graphs | 0.092 | 0.074 | 0.074 | 0.080 |
| engine + graphs, budget 256 | 0.087 | 0.081 | 0.086 | 0.095 |
| llama.cpp -ub 512 | 0.130 | 0.091 | 0.101 | 0.091 |
| llama.cpp -ub 2048 | 0.130 | 0.089 | 0.087 | 0.096 |
