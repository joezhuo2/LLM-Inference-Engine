Measured 2026-11-05T07:07:06Z at commit 1693c46: one request per run, median of 5 runs.

TTFT in ms against prompt tokens, median (min to max):

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine | 15.7 (15.7 to 15.8) | 97.8 (97.3 to 98.2) | 337.0 (334.8 to 338.2) | 1152.2 (1150.5 to 1161.9) |
| engine + graphs | 15.7 (15.6 to 15.7) | 97.8 (97.3 to 98.4) | 339.6 (338.1 to 342.0) | 1157.2 (1152.7 to 1184.1) |
| engine + graphs, budget 256 | 15.7 (15.6 to 15.9) | 101.1 (100.8 to 102.5) | 351.9 (351.3 to 353.5) | 1181.6 (1178.6 to 1195.6) |
| llama.cpp -ub 512 | 16.1 (14.9 to 17.1) | 44.0 (43.7 to 45.5) | 95.0 (94.8 to 95.7) | 179.5 (177.0 to 182.1) |
| llama.cpp -ub 2048 | 15.8 (15.4 to 17.7) | 44.0 (43.7 to 46.7) | 87.0 (86.0 to 87.5) | 185.3 (184.4 to 186.6) |

Median TTFT in ms per prompt token:

| system | 128 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| engine | 0.123 | 0.191 | 0.329 | 0.563 |
| engine + graphs | 0.123 | 0.191 | 0.332 | 0.565 |
| engine + graphs, budget 256 | 0.123 | 0.197 | 0.344 | 0.577 |
| llama.cpp -ub 512 | 0.126 | 0.086 | 0.093 | 0.088 |
| llama.cpp -ub 2048 | 0.124 | 0.086 | 0.085 | 0.090 |
