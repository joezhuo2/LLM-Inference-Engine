| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 107.6 | 9.21 | 9199.0 | 224.5 | 3 |
| engine | 2 | 191.6 | 10.05 | 4602.6 |  | 3 |
| engine | 4 | 399.4 | 9.57 | 2287.7 |  | 3 |
| engine | 8 | 665.2 | 11.07 | 1228.5 |  | 3 |
| engine | 16 | 1014.2 | 13.97 | 635.7 |  | 3 |
| HF transformers | 1 | 33.7 | 28.80 | 28728.1 | 71.8 | 1 |
| HF transformers | 4 | 115.6 | 32.73 | 7000.5 |  | 1 |
| HF transformers | 16 | 427.7 | 36.51 | 1183.1 |  | 1 |
