Measured 2026-11-10T12:33:36Z at commit 8297921: one request per run, median of 5 runs.

TTFT in ms against prompt tokens, median (min to max):

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine, paged prefill | 17.7 (16.9 to 17.9) | 120.6 (118.4 to 121.0) | 471.6 (469.5 to 472.7) | 1507.9 (1505.8 to 1515.3) |
| engine | 11.2 (11.2 to 11.5) | 46.4 (45.6 to 47.3) | 91.7 (91.6 to 94.8) | 203.1 (202.8 to 203.2) |
| engine + graphs | 11.2 (11.2 to 11.7) | 46.9 (45.5 to 47.3) | 94.6 (91.9 to 94.8) | 202.7 (201.9 to 203.4) |
| engine + graphs, budget 256 | 12.4 (11.9 to 13.3) | 52.7 (50.7 to 53.6) | 112.7 (108.6 to 114.2) | 253.3 (252.6 to 254.1) |
| llama.cpp -ub 512 | 17.2 (16.4 to 17.4) | 51.9 (51.5 to 52.8) | 112.4 (110.8 to 112.7) | 231.4 (223.6 to 234.9) |
| llama.cpp -ub 2048 | 17.2 (16.5 to 18.7) | 52.1 (51.2 to 52.2) | 103.0 (102.4 to 104.5) | 240.1 (236.6 to 244.5) |

Median TTFT in ms per prompt token:

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine, paged prefill | 0.139 | 0.236 | 0.461 | 0.736 |
| engine | 0.088 | 0.091 | 0.090 | 0.099 |
| engine + graphs | 0.088 | 0.092 | 0.092 | 0.099 |
| engine + graphs, budget 256 | 0.097 | 0.103 | 0.110 | 0.124 |
| llama.cpp -ub 512 | 0.135 | 0.101 | 0.110 | 0.113 |
| llama.cpp -ub 2048 | 0.135 | 0.102 | 0.101 | 0.117 |
