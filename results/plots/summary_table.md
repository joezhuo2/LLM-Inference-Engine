| system | max batch | output tok/s | TPOT p50 ms | TTFT p50 ms | weights GB/s | runs |
|---|---|---|---|---|---|---|
| engine | 1 | 107.6 | 9.20 | 9100.6 | 225.0 | 3 |
| engine | 2 | 189.6 | 10.84 | 4633.6 |  | 3 |
| engine | 4 | 396.0 | 9.65 | 2292.4 |  | 3 |
| engine | 8 | 658.3 | 11.29 | 1235.2 |  | 3 |
| engine | 16 | 1011.9 | 14.04 | 635.8 |  | 3 |
| engine + graphs | 1 | 117.9 | 8.43 | 8337.4 | 245.6 | 3 |
| engine + graphs | 4 | 426.5 | 8.91 | 2142.8 |  | 3 |
| engine + graphs | 16 | 1074.8 | 13.12 | 605.9 |  | 3 |
| HF transformers | 1 | 25.9 | 38.45 | 39002.3 | 53.8 | 1 |
| HF transformers | 4 | 105.2 | 38.17 | 8365.7 |  | 1 |
| HF transformers | 16 | 405.9 | 38.55 | 1347.4 |  | 1 |
