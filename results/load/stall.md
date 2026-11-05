Measured 2026-11-05T07:05:42Z at commit 1693c46: a 1917-token prompt arrives among short requests, medians of the runs.

| token budget | long prompt tokens | decoding when it arrives | TTFT of the long prompt ms | gap p99 ms | largest gap ms | runs |
|---|---|---|---|---|---|---|
| none | 1917 | 15 | 1030 | 76 | 1043 | 3 |
| 256 | 1917 | 15 | 1081 | 207 | 223 | 3 |
| 512 | 1917 | 15 | 1055 | 344 | 388 | 3 |
| 2048 | 1917 | 15 | 1040 | 14 | 1033 | 3 |
