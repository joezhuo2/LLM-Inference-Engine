# Branch 84: bench/chunked-prefill (issue #101, Part A)

Worktree `~/code/llm-inference-engine-bench`, branch `bench/chunked-prefill`, cut from `sched/chunked-prefill-wiring` with origin/main merged (8ac039c; its content equals main 2eb2f91 after #149 merged). Six local commits, unpushed, no PR. Each message is one line, a blank line, then `Co-Authored-By: Claude <noreply@anthropic.com>`, as the parent session asked (subagent_rules.md says no trailer; Joe's override wins).

## Commits

- 1693c46 Sweep token budgets and report inter-token gaps under load (load_sweep.sh TOKEN_BUDGET/RUNS/TRACES; load_report.py budget series, provenance line, --traces gap table and plot)
- a2d7e48 Time a token budget series in the prefill curve (prefill_ttft.py --token-budget)
- f5329ab Probe decode stalls when one long prompt arrives (scripts/stall_probe.py, auto:stall block in make_results.py)
- 1c88312 Measure chunked prefill under load and update the results (results/load, results/prefill, docs/results.md, docs/baselines.md, README, reproduce.sh)
- 5b317fc Show the token budget sweep against vLLM on the results page (web)
- 31bd307 Replace the unmeasured chunked prefill copy with the measured effect (web: story, sandbox, kernel timeline, home M6)

All Part A. No C++ changes.

## Measured (RTX 5060 Laptop GPU, release build, 2026-11-05, commit 1693c46, engine code = main 2eb2f91)

Workload as before: 120 requests, Poisson arrivals, prompts 16 to 512, outputs 16 to 256, seed 0, max 16 sequences, greedy, EOS ignored. 3 runs per point, budgets interleaved run by run. SLO TTFT 500 ms, TPOT 50 ms.

| QPS | goodput: whole / b256 / b512 / b2048 / vLLM | TTFT p99 ms: whole / b256 / b2048 / vLLM | TPOT p99 ms: whole / b256 / vLLM | largest gap ms: whole / b256 / b512 / b2048 |
|---|---|---|---|---|
| 1 | 1.14 / 1.14 / 1.14 / 1.14 / 1.14 | 102 / 121 / 106 / 68 | 11.2 / 10.3 / 8.5 | 158 / 65 / 97 / 96 |
| 2 | 2.24 / 2.24 / 2.24 / 2.24 / 2.25 | 115 / 136 / 120 / 48 | 12.9 / 12.1 / 8.1 | 161 / 65 / 98 / 100 |
| 3 | 3.30 / 3.30 / 3.30 / 3.30 / 3.33 | 123 / 146 / 130 / 39 | 15.6 / 14.0 / 8.3 | 178 / 66 / 98 / 102 |
| 4 | 4.21 / 4.28 / 4.28 / 4.25 / 4.36 | 605 / 413 / 506 / 41 | 17.8 / 16.2 / 8.4 | 201 / 67 / 100 / 116 |
| 5 | 4.42 / 4.57 / 4.50 / 4.44 / 5.36 | 1192 / 1033 / 1105 / 55 | 17.2 / 16.4 / 8.7 | 212 / 68 / 103 / 163 |
| 6 | 3.66 / 4.05 / 3.97 / 3.96 / 6.34 | 1991 / 1642 / 1761 / 90 | 19.7 / 17.5 / 8.7 | 182 / 70 / 105 / 158 |
| 8 | 1.50 / 1.61 / 1.60 / 1.61 / 8.07 | 5007 / 4316 / 4397 / 507 | 19.9 / 18.6 / 8.8 | 173 / 71 / 106 / 160 |

Stall probe (one 1,917-token prompt arriving while 15 short requests decode, 3 runs): largest gap 1,043 ms (no budget), 223 (256), 388 (512), 1,033 (2,048); the long prompt's TTFT 1,030 / 1,081 / 1,055 / 1,040 ms.

Prefill alone, budget 256 vs whole (graphs): 2,048 tokens 1,181.6 vs 1,157.2 ms (+2.1%), +3 to 4% at 512 and 1,024.

Load average (1-minute): median over each engine run between 0.99 and 4.75 (median of runs 1.6, highest sample 5.25, about 0.7 of it the engine itself); vLLM runs median 1.3, max 2.8; stall and prefill runs about 1.1 to 1.6. No gpu_tests ran during any run (sampled every 5 s). Outliers: none; the largest TPOT p50 deviation from its repeats' median was 3.5% (budget 256, QPS 6, run 1, faster than its repeats), vLLM within 1%. Nothing redone.

Why vLLM was rerun: the 2026-11-03 sweep is not comparable with today. The same 2fec1f3 binary it used, rerun today at QPS 4 and 6, gives TPOT p50 12.26 and 15.30 ms (new build without traces 12.36 and 15.28, with traces 12.32 and 15.27 median), against 18.5 and 19.6 ms on 11-03. Same code, so the machine was slower on 11-03 (GPU clock not logged, cause unknown). vLLM also moved (TPOT p50 at QPS 4: 10.4 to 8.1 ms), and the prefill curve moved for both systems (engine 1,612 to 1,152 ms, llama.cpp 265 to 180 ms at 2,048). The new results supersede the 11-03 files of the same names.

## Honest reading

- Chunking does what it is for: the longest decode stall drops about 3x at budget 256 (201 to 67 ms at QPS 4; 1,043 to 223 ms in the stall probe) and the TTFT tail improves (605 to 413 ms p99 at QPS 4). 2,048 helps little on this workload because prompts up to 512 still fit whole into a step.
- Goodput barely moves: peak 4.42 to 4.57 req/s (+3%), better past the peak (3.66 to 4.05 at QPS 6). vLLM reaches 6.34 at 6 and 8.07 at 8, about 1.8x the engine's best. Stalls were only part of the gap.
- Chunking moves prefill work, it does not shrink it. In the whole-prompt QPS 8 run, 114 prefill steps take 5.3 of 19.7 s (0.17 ms per prompt token) while decode steps average 10.5 ms, close to vLLM's 8.5 ms TPOT with prefill included. Each chunk row still runs the paged decode attention over its context, so a 241-row chunk late in a 1,917-token prompt still costs over 200 ms. The rest of the gap is prefill kernel speed (#103), not scheduling. Mixed steps also run eager (no graph).
- Small budgets cost a little at low load: budget 256 adds 6 to 10 ms TTFT p50 at 1 to 3 QPS and raises gap p99 at 1 QPS from 9.6 to 26.4 ms. 512 costs almost nothing there; I recommend 512 as a default for this workload and 256 for the shortest stalls.
- TPOT p99 hides stalls (per-request mean): under 20 ms for every engine configuration.

## What was updated where

- scripts: load_sweep.sh, load_report.py, prefill_ttft.py, stall_probe.py (new), make_results.py (auto:stall), reproduce.sh (all budgets, traces in build/load_traces, stall probe, prefill budget 256). Tests: test_load_report.py, test_prefill_ttft.py, test_stall_probe.py (new), test_make_results.py.
- results/load: engine_q*, engine_budget{256,512,2048}_q*, vllm_q* (rerun), token_gaps.csv, stall.csv, stall.md, load_table.md, goodput/ttft/tpot plots and gap_max_vs_qps.png. results/prefill: rerun with the budget series and llama.cpp.
- docs/results.md "Under load" (method, GPU, commit, date, load averages, outlier check, why 11-03 is superseded, reading, limits), new "A long prompt among running requests", "Prefill against prompt length" prose with new numbers and the budget row. docs/baselines.md vLLM bullet and reading table. README continuous batching line and "Beyond the MVP" (chunked prefill removed).
- web: sync-data.mjs reads the "Engine runs measured ... at commit ... on ..." provenance line (loadRun()); results story has two new steps (goodput vs QPS; TTFT p99 log axis with SLO line and largest-gap stats) with engine whole / budget 256 / budget 2,048 / vLLM; results intro, description and home card mention vLLM. Chunked prefill story: headline is now the measured stall 1,043 to 223 ms, stall and spread steps quote the sweep and probe, testing text, sources/tests, measurements (load tables 0 and 1, stall table), timeline (#149 wiring, this branch measured; the 11-03 vLLM entry no longer quotes numbers). Sandbox "for scale" note adds the measured stall. Home M6 entry: #133 #134 #149, measured text, bars of the longest gap at 4 QPS.

## Verification

- Engine gate (`~/.cache/llmie-bench-chunked/verify.sh`, a copy of the chunked one pointed at this worktree) under `flock ~/.cache/llmie-bench/gpu.lock` for commits 1 to 4: GPU up, debug 701/701, cpu 313/313, VERIFY OK each time. Python: `python3 -m unittest discover -s tests/python` (skips matplotlib tests) and `~/.venvs/llmie/bin/python` (runs them), 95, 97, 102, 102 tests, OK, run in a temporary worktree holding exactly the commit. Web gate (`verify-web.sh` pointed at the temp worktree, cp -al node_modules) for commits 4, 5 and 6: VERIFY WEB OK. Logs: ~/.cache/llmie-bench-chunked/gate-c*.log.
- Since no commit touches C++, the engine build and ctest ran in the main worktree whose C++ equals every commit's; the format check, python tests and web gate ran on the exact commit content.
- Browser: headless Chrome on 9490 over CDP, static build on 8490: /results (both new steps), / (M6 entry), /kernels/chunked-prefill (headline, steps 1 and 3, timeline entry, sandbox note) at 1280 and 390, dark and light: every target found, 0 overflowing elements, 0 console errors or exceptions. Screenshots in ~/.cache/llmie-bench-chunked/shots/final_*.png. Fixed on the way: the goodput axis cut vLLM off at 7 (now derived from the data), a wrong "at least every 50 ms" phrasing, unformatted numbers over 1,000, and "--token-budget" breaking across lines at 390 px.

## Not verified

- Full `scripts/reproduce.sh` end to end (it now takes much longer: 4 budgets x 7 rates x 3 runs plus the stall probe; about 90 minutes for the sweep alone).
- Reduced-motion screenshots and animation sampling of the new chart steps (they reuse the existing LineChart, no new motion).
- The sync-data provenance regex has no unit test (same as the existing ceilings regex).
- GPU clock during the runs was not logged, so the cause of the 11-03 slowdown is unknown.
- Commit labels: the results say commit 1693c46 (HEAD when measured; the engine binary was built from 8ac039c, identical C++).

## Open questions for Joe

1. Default budget: the site and docs headline budget 256 (shortest stalls); 512 is nearly as good under load and cheaper at low load. Should `engine serve` get a default budget (512?) or stay 0?
2. The 11-03 numbers quoted in status.md and older summaries (3.2 vs 6.1 req/s) are superseded; the vLLM-to-engine ratio is now about 1.8x at the engine's peak and the engine re-measured faster on the same code. OK to treat the 11-03 sweep as discarded?
3. Traces are not committed (about 1 to 3 MB each, 84 runs); token_gaps.csv is. Fine?
4. Next fix for serving is prefill attention (#103): the traces show prefill at 27% of the QPS 8 run.

## Draft PR description

Title: Measure chunked prefill under load (#101)

Measures what `--token-budget` (#149) does to serving, and updates the docs and site.

- `scripts/load_sweep.sh` takes `TOKEN_BUDGET`, `RUNS` and `TRACES`; `scripts/load_report.py` reports budgets as their own series, writes a provenance line, and with `--traces` adds the gaps between tokens (p50, p99, largest) and a plot.
- `scripts/stall_probe.py`: one 1,917-token prompt arriving among 15 decoding requests, per budget. `scripts/prefill_ttft.py --token-budget` adds a chunked series to the prefill curve. `scripts/reproduce.sh` runs all of it.
- Re-measured on 2026-11-05 at budgets 0, 256, 512 and 2,048 (3 runs, interleaved), with vLLM rerun the same day: the 11-03 numbers came from a slower machine state (the same binary is 35% faster today).
- Result: budget 256 cuts the longest decode stall about 3x (201 to 67 ms at 4 QPS; 1,043 to 223 ms when a long prompt arrives) and TTFT p99 at 4 QPS from 605 to 413 ms, but peak goodput only from 4.42 to 4.57 req/s against vLLM's 8.07. Chunking moves prefill work and does not shrink it; the remaining gap is prefill kernel speed (#103).
- docs/results.md, docs/baselines.md and README updated; the results page gets goodput and tail-latency steps with the budget series and vLLM; the chunked prefill story, its sandbox and the home timeline now show the measured effect.
