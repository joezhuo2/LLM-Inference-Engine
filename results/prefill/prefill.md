Measured 2026-11-03T13:48:36Z at commit 2fec1f3: one request per run, median of 5 runs.

TTFT in ms against prompt tokens, median (min to max):

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine | 18.3 (17.2 to 20.8) | 123.5 (118.4 to 126.6) | 456.7 (446.6 to 498.9) | 1611.6 (1591.7 to 1844.6) |
| engine + graphs | 19.9 (18.2 to 21.6) | 119.1 (118.2 to 127.9) | 471.3 (443.5 to 569.0) | 1593.2 (1555.1 to 1867.4) |
| llama.cpp -ub 512 | 19.5 (17.1 to 40.8) | 62.3 (53.8 to 100.8) | 129.2 (116.1 to 151.4) | 264.5 (234.2 to 279.3) |
| llama.cpp -ub 2048 | 16.2 (15.1 to 17.9) | 52.1 (50.9 to 54.8) | 104.0 (101.9 to 114.4) | 271.0 (261.6 to 302.1) |

Median TTFT in ms per prompt token:

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine | 0.143 | 0.241 | 0.446 | 0.787 |
| engine + graphs | 0.155 | 0.233 | 0.460 | 0.778 |
| llama.cpp -ub 512 | 0.152 | 0.122 | 0.126 | 0.129 |
| llama.cpp -ub 2048 | 0.127 | 0.102 | 0.102 | 0.132 |
