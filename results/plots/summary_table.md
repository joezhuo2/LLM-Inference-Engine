| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 107.3 | 9.28 | 9189.4 | 222.9 | 3 |
| engine | 2 | 188.7 | 10.71 | 4670.7 |  | 3 |
| engine | 4 | 389.9 | 9.82 | 2341.9 |  | 3 |
| engine | 8 | 656.3 | 11.21 | 1251.9 |  | 3 |
| engine | 16 | 992.7 | 14.28 | 648.4 |  | 3 |
| engine + graphs | 1 | 117.0 | 8.49 | 8428.6 | 243.6 | 3 |
| engine + graphs | 4 | 422.4 | 8.95 | 2174.9 |  | 3 |
| engine + graphs | 16 | 1059.6 | 13.25 | 617.2 |  | 3 |
| HF transformers | 1 | 33.4 | 28.97 | 29320.9 | 71.4 | 1 |
| HF transformers | 4 | 130.2 | 30.54 | 6984.5 |  | 1 |
| HF transformers | 16 | 487.3 | 31.83 | 1167.3 |  | 1 |
