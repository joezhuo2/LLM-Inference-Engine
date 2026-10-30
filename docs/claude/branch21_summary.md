# Branch 21: tools/bench

Base: origin/main 37a4712 (#29). Four commits, each passing the gate alone (format, debug and cpu builds without warnings, `ctest -LE todo_joe`, GPU tests included). All Part A.

| Commit | Content |
| --- | --- |
| cfafce5 Add synthetic bench workloads with Poisson arrivals | `make_workload`: BOS plus random non-special tokens, lengths `N` or `MIN:MAX`, arrivals all at once or Poisson at `--request-rate`, SplitMix64 so prompts match across standard libraries. 11 CPU tests. |
| d01384d Add bench metrics with percentiles and CSV output | `request_metrics` (TTFT, TPOT, latency per the build plan), `summarize` (throughput, mean/p50/p90/p99), numpy-style percentiles, CSV writers. 12 CPU tests. |
| 3ae0866 Add a bench driver running requests one at a time | `run_one_at_a_time` serves a workload through `generate` at batch 1, honoring arrivals, EOS optional. 6 GPU tests. |
| 523de4f Add engine bench writing per-request and summary CSVs | `engine bench` command, `ENGINE_BUILD_TYPE` definition, README Running section, `results/` in the layout. |

## Usage

```bash
./build/release/bin/engine bench --model models/tinyllama --num-requests 64 --prompt-len 128 --output-len 256 --out results/laptop_b1.csv
```

Writes `results/laptop_b1.csv` (one row per request: `id,prompt_tokens,output_tokens,preemptions,arrival_s,ttft_s,tpot_s,latency_s`) and `results/laptop_b1_summary.csv` (settings, throughput, TTFT/TPOT/latency stats, `decode_weight_bytes`, `bandwidth_gb_s`).

## Decisions (confirmed by Joe on Oct 11)

- Summary in its own `<out>_summary.csv`, settings first so every result says how it was made.
- EOS ignored by default (`--stop-at-eos` to stop), warmup 3 untimed requests, request `i` samples with `seed + i`.
- Batch 1 until branch 20 adds the paged runner; `--max-batch`, `--graphs` and peak KV blocks come then.
- One-token requests have no TPOT and are left out of the TPOT stats; percentiles interpolate linearly like numpy.
- Achieved bandwidth = weights read per decode step (2,069,024,768 bytes, embedding table excluded) / TPOT p50.

## Measured (RTX 5060, release, Oct 11)

8 requests of 128 prompt and 128 output tokens: 10.67 s, 96 tokens/s, TPOT p50 10.36 ms (200 GB/s of weights), TTFT 25 ms for the first request and 9.3 s for the last (queued behind seven). `generate` ran at 103 tokens/s right after, so the README's 125 to 135 reflects the laptop's state at the time, not bench overhead.

## Worth a look

- `src/runtime/bench.cpp`: where the timestamps come from.
- `src/app/bench_command.cpp`: the weight-bytes formula.
