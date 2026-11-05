#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/load_sweep.sh [--quick]

Serves Poisson arrivals of mixed-length requests through engine bench at each QPS point and writes
results/load/engine_q<rate>_r<run>.csv and its _summary.csv, or engine_budget<N>_q<rate>_r<run>.csv
with TOKEN_BUDGET=N (chunked prefill, engine bench --token-budget N).

Environment: ENGINE (default build/release/bin/engine), MODEL (default models/tinyllama), RATES
(default "1 2 3 4 5 6 8"), REPEATS (3), RUNS (run numbers, default 1 to REPEATS), REQUESTS (120),
PROMPT_LEN (16:512), OUTPUT_LEN (16:256), MAX_BATCH (16), TOKEN_BUDGET (0, whole-prompt prefill),
TRACES (a directory for one step trace per run, none by default), OUT (default results/load).
USAGE
    exit 2
}

rates=${RATES:-1 2 3 4 5 6 8}
repeats=${REPEATS:-3}
requests=${REQUESTS:-120}
for arg in "$@"; do
    case "$arg" in
    --quick)
        rates="2 4"
        repeats=1
        requests=16
        ;;
    *) usage ;;
    esac
done

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
prompt_len=${PROMPT_LEN:-16:512}
output_len=${OUTPUT_LEN:-16:256}
max_batch=${MAX_BATCH:-16}
budget=${TOKEN_BUDGET:-0}
runs=${RUNS:-$(seq 1 "$repeats")}
traces=${TRACES:-}
out=${OUT:-results/load}
mkdir -p "$out"
[ -n "$traces" ] && mkdir -p "$traces"

name=engine
[ "$budget" -gt 0 ] && name=engine_budget$budget
for r in $runs; do
    for rate in $rates; do
        echo "$name qps $rate run $r"
        run=${name}_q${rate}_r$r
        extra=()
        [ "$budget" -gt 0 ] && extra+=(--token-budget "$budget")
        [ -n "$traces" ] && extra+=(--trace "$traces/$run.json")
        "$engine" bench --model "$model" --max-batch "$max_batch" --graphs \
            --num-requests "$requests" --prompt-len "$prompt_len" --output-len "$output_len" \
            --request-rate "$rate" --seed 0 --out "$out/$run.csv" "${extra[@]}" > /dev/null
    done
done
