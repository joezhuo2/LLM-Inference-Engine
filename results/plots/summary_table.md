| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 114.0 | 8.61 | 8766.2 | 240.4 | 3 |
| engine | 2 | 222.8 | 8.79 | 4233.9 |  | 3 |
| engine | 4 | 432.9 | 8.74 | 2147.2 |  | 3 |
| engine | 8 | 765.7 | 9.39 | 1102.2 |  | 3 |
| engine | 16 | 1283.0 | 10.81 | 521.8 |  | 3 |
| engine + graphs | 1 | 126.7 | 7.83 | 7786.0 | 264.2 | 3 |
| engine + graphs | 4 | 473.0 | 7.93 | 1949.9 |  | 3 |
| engine + graphs | 16 | 1405.8 | 9.67 | 497.0 |  | 3 |
| HF transformers | 1 | 24.0 | 40.96 | 44404.9 | 50.5 | 1 |
| HF transformers | 4 | 89.6 | 45.80 | 9969.9 |  | 1 |
| HF transformers | 16 | 345.4 | 45.40 | 1590.1 |  | 1 |
