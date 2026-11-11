Measured 2026-11-11T03:55:32Z at commit 1feaba9: a 1917-token prompt arrives among short requests, medians of the runs.

| token budget | long prompt tokens | decoding when it arrives | TTFT of the long prompt ms | gap p99 ms | largest gap ms | runs |
|---|---|---|---|---|---|---|
| none | 1917 | 15 | 230 | 60 | 241 | 3 |
| 256 | 1917 | 15 | 297 | 37 | 38 | 3 |
| 512 | 1917 | 15 | 267 | 66 | 67 | 3 |
| 2048 | 1917 | 15 | 234 | 18 | 229 | 3 |
