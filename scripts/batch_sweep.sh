#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/batch_sweep.sh [--quick]

Runs engine bench past max batch 16 (#121), eager and with CUDA Graphs, with twice as many
requests as the max batch so the batch fills twice, and writes results/batch/engine_b<N>_r<run>.csv
and engine_graphs_b<N>_r<run>.csv with their _summary.csv. Then the padding runs: PAD_REQUESTS
requests at once with --max-batch PAD_BATCH and --graphs, once per --graph-steps value, written to
results/batch/pad_steps<S>_r<run>.csv.

Environment: ENGINE (default build/release/bin/engine), MODEL (default models/tinyllama), BATCHES
(default "16 32 64 128 256"), REPEATS (3), PROMPT_LEN (64), OUTPUT_LEN (64), PAD_REQUESTS (17),
PAD_BATCH (32), PAD_STEPS (default "1 2 4 8"), PAD_OUTPUT_LEN (256), PAD_REPEATS (5), SKIP_SWEEP
(1 runs only the padding runs), OUT (default results/batch).
USAGE
    exit 2
}

batches=${BATCHES:-16 32 64 128 256}
repeats=${REPEATS:-3}
pad_steps=${PAD_STEPS:-1 2 4 8}
for arg in "$@"; do
    case "$arg" in
    --quick)
        batches="16 32"
        repeats=1
        pad_steps="1 8"
        pad_repeats=1
        ;;
    *) usage ;;
    esac
done

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
prompt_len=${PROMPT_LEN:-64}
output_len=${OUTPUT_LEN:-64}
pad_requests=${PAD_REQUESTS:-17}
pad_batch=${PAD_BATCH:-32}
pad_output_len=${PAD_OUTPUT_LEN:-256}
pad_repeats=${PAD_REPEATS:-5}
skip_sweep=${SKIP_SWEEP:-0}
out=${OUT:-results/batch}
mkdir -p "$out"

bench() {
    "$engine" bench --model "$model" --prompt-len "$prompt_len" --seed 0 "$@" > /dev/null
}

if [ "$skip_sweep" != 1 ]; then
    for r in $(seq 1 "$repeats"); do
        for b in $batches; do
            echo "batch $b run $r"
            bench --max-batch "$b" --num-requests $((2 * b)) --output-len "$output_len" --out "$out/engine_b${b}_r$r.csv"
            bench --max-batch "$b" --num-requests $((2 * b)) --output-len "$output_len" --graphs \
                --out "$out/engine_graphs_b${b}_r$r.csv"
        done
    done
fi
for r in $(seq 1 "$pad_repeats"); do
    for s in $pad_steps; do
        echo "padding: $pad_requests requests, max batch $pad_batch, graph steps $s, run $r"
        bench --max-batch "$pad_batch" --num-requests "$pad_requests" --output-len "$pad_output_len" \
            --graphs --graph-steps "$s" --out "$out/pad_steps${s}_r$r.csv"
    done
done
