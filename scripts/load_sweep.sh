#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/load_sweep.sh [--quick]

Serves Poisson arrivals of mixed-length requests through engine bench at each QPS point and writes
results/load/engine_q<rate>_r<run>.csv and its _summary.csv.

Environment: ENGINE (default build/release/bin/engine), MODEL (default models/tinyllama), RATES
(default "1 2 3 4 5 6 8"), REPEATS (3), REQUESTS (120), PROMPT_LEN (16:512), OUTPUT_LEN (16:256),
MAX_BATCH (16), OUT (default results/load).
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
out=${OUT:-results/load}
mkdir -p "$out"

for r in $(seq 1 "$repeats"); do
    for rate in $rates; do
        echo "qps $rate run $r"
        "$engine" bench --model "$model" --max-batch "$max_batch" --graphs \
            --num-requests "$requests" --prompt-len "$prompt_len" --output-len "$output_len" \
            --request-rate "$rate" --seed 0 --out "$out/engine_q${rate}_r$r.csv" > /dev/null
    done
done
