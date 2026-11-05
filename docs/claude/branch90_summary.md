# Branch 90: tools/onchip-speeds

Worktree `~/code/llm-inference-engine-speeds`, branch `tools/onchip-speeds` from origin/main d13ff65. 4 local commits, not pushed.

## Commits
- e608146 Add an on-chip memory microbenchmark and its report script (scripts/onchip_bench.cu, scripts/onchip_report.py, tests/python/test_onchip_report.py, reproduce.sh `--skip-onchip`, make_results `auto:onchip` block + test)
- 1d2e18b Record L2, shared memory, register and latency numbers in results (results/onchip/onchip.csv + onchip.md, docs/results.md "On-chip memory", docs/hardware.md register line)
- 1c9ce9f Show measured L2, shared memory and register speeds in foundations (sync-data onchip.json, onchip()/onchipRun(), memory-tier figure, copy, caption, measurements table)
- e52676c Quote measured load latencies in the split attention story

## Measured (RTX 5060 Laptop GPU, commit e608146, 2026-11-05T04:40Z, 5 runs, median (min to max), SM clock from clock64 vs %globaltimer inside each kernel)
- L2 read, 8 MB, __ldcg: 1,911 GB/s (1,905 to 1,913), 26.3 B/clk/SM, 2,790 MHz
- L2 hit latency, chase in 4 MB: 318 cycles (318 to 318), 113 ns
- shared memory 128-bit: 9,107 GB/s (9,097 to 9,115), 350 GB/s/SM, 125 B/clk/SM, 2,800 MHz
- shared memory 32-bit: 4,644 GB/s (4,644 to 4,645), 179 GB/s/SM, 63.7 B/clk/SM
- shared memory latency: 28.0 cycles, 9.96 ns, 2,812 MHz
- dependent FMA: 4.05 cycles, 1.44 ns
- DRAM latency, chase in 256 MB: 867 cycles (866 to 915), 309 ns (308 to 326), 2,809 MHz
- register file: 256 KB/SM, 6,656 KB total (regsPerMultiprocessor)
- Load: 1.94 at start, 2.58 at end (10 cores); no gpu_tests during the run (polled every 0.2 s); nvidia-smi before: SM 1,687 MHz idle, mem 12,001 MHz, P0.
- Earlier exploratory session (load 3.2 to 3.8): same within 2%, DRAM latency 864 to 915. A first run that overlapped another agent's gpu_tests gave L2 1,261 GB/s and DRAM 1,012 cycles and was discarded.

## Caveats
- 32-bit shared loads at exactly half of 128-bit: unchanged with 8 or 16 loads per iteration and 128-thread blocks, so not the loop; the reason (LSU taking 2 cycles per warp-wide 32-bit load) is an inference, documented as such.
- DRAM chase over 256 MB may include address-translation misses; one run of 5 at 915 cycles.
- B/clk/SM uses SM cycles; L2 has its own clock domain. L2 vs DRAM ratio in the docs is stated against both the 323.6 copy ceiling and an exploratory read sweep (359/372 GB/s at 64/256 MB, not in the table).
- No register bandwidth number: only file size and the 4-cycle FMA dependency.

## Gates
- e608146: engine gate VERIFY OK (debug 694 tests, cpu 309), run before the commit on the same staged tree, then committed (not chained with &&; deviation).
- 1d2e18b: engine gate && commit, VERIFY OK (debug 694, cpu 309).
- 1c9ce9f and e52676c: web gate && commit, VERIFY WEB OK each, each tree alone (split-attention change set aside for 1c9ce9f).
- Python: 96 tests OK with python3 (4 matplotlib skips) and the venv. Web npm test 22/22.
- Browser (headless Chrome 9486, static build 8486): foundations memory step at 1280 and 390, dark and light, reduced motion, no console errors, no SVG text overflow; cdp-phone step sync ok at 390 and 1280 (an earlier longer copy desynced at one scroll position on phone, fixed by tightening the copy); split-attention step 2 checked; on-chip table renders.

## Not verified
- Full scripts/reproduce.sh end to end (only the on-chip section's commands were run by hand).
- compute-sanitizer on onchip_bench.
- 1d2e18b's tree was not web-built alone (it only adds a docs section).
- Joe's next dev on port 3000 (other worktree/branch).
- Animation sampling: the figure keeps the existing motion, not re-sampled.

## Open questions
- Commit trailer: subagent_rules.md says no trailer, the launch brief said Co-Authored-By; commits follow the brief.
- reproduce.sh runs the on-chip section by default (about a minute) with -arch=native; fine?
- docs/hardware.md now has a hand-typed "registers per SM: 256 KB" line (also in the CSV); keep or drop?
