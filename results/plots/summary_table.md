| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 114.3 | 8.66 | 8699.5 | 239.0 | 3 |
| engine | 2 | 209.1 | 9.25 | 4310.7 |  | 3 |
| engine | 4 | 426.4 | 8.96 | 2119.1 |  | 3 |
| engine | 8 | 757.3 | 9.61 | 1056.4 |  | 3 |
| engine | 16 | 1282.8 | 10.80 | 518.7 |  | 3 |
| engine + graphs | 1 | 124.4 | 7.96 | 7929.6 | 260.1 | 3 |
| engine + graphs | 4 | 468.4 | 8.07 | 1965.2 |  | 3 |
| engine + graphs | 16 | 1386.2 | 9.83 | 501.6 |  | 3 |
| HF transformers | 1 | 24.0 | 40.96 | 44404.9 | 50.5 | 1 |
| HF transformers | 4 | 89.6 | 45.80 | 9969.9 |  | 1 |
| HF transformers | 16 | 345.4 | 45.40 | 1590.1 |  | 1 |
