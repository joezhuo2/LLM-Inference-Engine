SLO: TTFT at most 500 ms and TPOT at most 50 ms; goodput is the requests per second that meet both.

| system | QPS | output tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | SLO met | goodput req/s | preemptions | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| engine + graphs | 1 | 167 | 64 | 162 | 11.0 | 14.7 | 100% | 1.13 | 0 | 3 |
| engine + graphs | 2 | 326 | 62 | 166 | 12.7 | 18.6 | 100% | 2.22 | 0 | 3 |
| engine + graphs | 3 | 477 | 84 | 777 | 15.9 | 23.0 | 93% | 3.03 | 0 | 3 |
| engine + graphs | 4 | 605 | 121 | 1653 | 18.5 | 23.4 | 78% | 3.22 | 0 | 3 |
| engine + graphs | 5 | 667 | 591 | 3328 | 19.7 | 25.8 | 48% | 2.15 | 0 | 3 |
| engine + graphs | 6 | 685 | 1522 | 5973 | 19.8 | 25.8 | 25% | 1.16 | 0 | 3 |
| engine + graphs | 8 | 710 | 3027 | 9389 | 20.1 | 25.4 | 23% | 1.13 | 0 | 3 |
