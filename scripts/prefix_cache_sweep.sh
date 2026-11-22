#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/prefix_cache_sweep.sh [--quick]

Serves requests that all start with the same random prefix (a system prompt) through engine bench,
with the prefix cache on and with --no-prefix-cache, and writes results/prefix/<on|off>_p<prefix>_q<rate>_r<run>.csv
and its _summary.csv, one step trace per run in TRACES; scripts/prefix_cache_report.py makes the table.

Environment: ENGINE (default build/release/bin/engine), MODEL (default models/tinyllama), PREFIXES
(default "256 512 1024"), RATES (default "2 0", 0 submits every request at once), REPEATS (3),
REQUESTS (32), PROMPT_LEN (64, the tokens of each request after the prefix), OUTPUT_LEN (32),
MAX_BATCH (16), TRACES (default build/prefix_traces), OUT (default results/prefix).
USAGE
    exit 2
}

prefixes=${PREFIXES:-256 512 1024}
rates=${RATES:-2 0}
repeats=${REPEATS:-3}
requests=${REQUESTS:-32}
for arg in "$@"; do
    case "$arg" in
    --quick)
        prefixes=512
        rates=2
        repeats=1
        requests=8
        ;;
    *) usage ;;
    esac
done

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
prompt_len=${PROMPT_LEN:-64}
output_len=${OUTPUT_LEN:-32}
max_batch=${MAX_BATCH:-16}
traces=${TRACES:-build/prefix_traces}
out=${OUT:-results/prefix}
mkdir -p "$out" "$traces"

for r in $(seq 1 "$repeats"); do
    for prefix in $prefixes; do
        for rate in $rates; do
            for cache in on off; do
                run=${cache}_p${prefix}_q${rate}_r$r
                echo "$run"
                extra=()
                [ "$cache" = off ] && extra+=(--no-prefix-cache)
                "$engine" bench --model "$model" --max-batch "$max_batch" --graphs \
                    --num-requests "$requests" --shared-prefix "$prefix" --prompt-len "$prompt_len" \
                    --output-len "$output_len" --request-rate "$rate" --seed 0 \
                    --out "$out/$run.csv" --trace "$traces/$run.json" "${extra[@]}" > /dev/null
            done
        done
    done
done
