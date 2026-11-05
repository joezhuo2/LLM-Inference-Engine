Measured 2026-11-05T04:40:03Z at commit e608146 on the NVIDIA GeForce RTX 5060 Laptop GPU: 5 runs of `scripts/onchip_bench.cu`, median (min to max). The SM clock is measured inside each kernel (`clock64()` against `%globaltimer`).

| tier | measure | unit | median | min to max | SM clock MHz |
|---|---|---|---|---|---|
| registers | register file | KB/SM | 256 | 256 to 256 |  |
| registers | register file | KB | 6,656 | 6,656 to 6,656 |  |
| shared memory | size | KB/SM | 100 | 100 to 100 |  |
| L2 | size | KB | 32,768 | 32,768 to 32,768 |  |
| shared memory | read bandwidth, 32-bit loads | GB/s | 4,644 | 4,644 to 4,645 | 2,804 |
| shared memory | read bandwidth, 32-bit loads | GB/s/SM | 179 | 179 to 179 | 2,804 |
| shared memory | read bandwidth, 32-bit loads | B/clk/SM | 63.7 | 63.7 to 63.7 | 2,804 |
| shared memory | read bandwidth, 128-bit loads | GB/s | 9,107 | 9,097 to 9,115 | 2,800 |
| shared memory | read bandwidth, 128-bit loads | GB/s/SM | 350 | 350 to 351 | 2,800 |
| shared memory | read bandwidth, 128-bit loads | B/clk/SM | 125 | 125 to 125 | 2,800 |
| L2 | read bandwidth, 8 MB working set | GB/s | 1,911 | 1,905 to 1,913 | 2,790 |
| L2 | read bandwidth, 8 MB working set | B/clk/SM | 26.3 | 26.3 to 26.4 | 2,790 |
| registers | dependent FMA latency | cycles | 4.05 | 4.05 to 4.05 | 2,812 |
| registers | dependent FMA latency | ns | 1.44 | 1.44 to 1.44 | 2,812 |
| shared memory | load latency, pointer chase | cycles | 28.0 | 28.0 to 28.0 | 2,812 |
| shared memory | load latency, pointer chase | ns | 9.96 | 9.96 to 9.96 | 2,812 |
| L2 | hit latency, pointer chase in 4 MB | cycles | 318 | 318 to 318 | 2,812 |
| L2 | hit latency, pointer chase in 4 MB | ns | 113 | 113 to 113 | 2,812 |
| DRAM | load latency, pointer chase in 256 MB | cycles | 867 | 866 to 915 | 2,809 |
| DRAM | load latency, pointer chase in 256 MB | ns | 309 | 308 to 326 | 2,809 |
