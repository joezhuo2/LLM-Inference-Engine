#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/speculative_more.sh [--quick]

The measurements scripts/speculative_sweep.sh leaves out, each with engine bench --max-batch
MAX_BATCH --graphs --stop-at-eos on the repeat prompts (scripts/prompts_repeat.json) and the first
CHAT_PROMPTS chat prompts (scripts/prompts.json), written as results/speculative_more/<part>_<set>_<variant>_r<run>.csv
with their _summary.csv, for scripts/speculative_more_report.py:

  running  SEQS requests arriving together, so that many sequences decode at once and share the rows
  load     LOAD_REQUESTS requests arriving at RATES requests per second
  sampled  one request at a time at TEMPERATURE and TOP_P instead of greedy
  draft    one request at a time with the guesses of DRAFT_MODEL (skipped when it is missing)

Every part runs without guesses (k0), with up to DRAFT guesses (k8) and with the adaptive limit
(k8a); the draft part runs DRAFT_LENGTHS guesses of the draft model (d2, d4, d8).

Environment: ENGINE (default build/release/bin/engine), MODEL (models/tinyllama), DRAFT_MODEL
(models/stories110m), SEQS ("1 2 4 8 12 16"), RATES ("2 4 8"), LOAD_REQUESTS (64), DRAFT (8),
DRAFT_LENGTHS ("2 4 8"), TEMPERATURE (0.7), TOP_P (0.9), REPEATS (3), OUTPUT_LEN (256),
MAX_BATCH (16), CHAT_PROMPTS (6), OUT (results/speculative_more), RUN_TIMEOUT (600 seconds a run).

A run whose summary is already in OUT is skipped, so a stopped sweep picks up where it stopped.
USAGE
    exit 2
}

seqs=${SEQS:-1 2 4 8 12 16}
rates=${RATES:-2 4 8}
repeats=${REPEATS:-3}
for arg in "$@"; do
    case "$arg" in
    --quick)
        seqs="1 4"
        rates="4"
        repeats=1
        ;;
    *) usage ;;
    esac
done

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
draft_model=${DRAFT_MODEL:-models/stories110m}
load_requests=${LOAD_REQUESTS:-64}
draft=${DRAFT:-8}
draft_lengths=${DRAFT_LENGTHS:-2 4 8}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.9}
output_len=${OUTPUT_LEN:-256}
max_batch=${MAX_BATCH:-16}
chat_prompts=${CHAT_PROMPTS:-6}
out=${OUT:-results/speculative_more}
run_timeout=${RUN_TIMEOUT:-600}
root=$(cd "$(dirname "$0")/.." && pwd)
sets=$(mktemp -d)
trap 'rm -rf "$sets"' EXIT
mkdir -p "$out"

cp "$root/scripts/prompts_repeat.json" "$sets/repeat.json"
python3 -c 'import json, sys; json.dump({"prompts": json.load(open(sys.argv[1]))["prompts"][:int(sys.argv[2])]}, open(sys.argv[3], "w"))' \
    "$root/scripts/prompts.json" "$chat_prompts" "$sets/chat.json"

bench() {
    local name=$1
    shift
    if [ -s "$out/${name}_summary.csv" ]; then return; fi
    echo "$name"
    timeout "$run_timeout" "$engine" bench --model "$model" --max-batch "$max_batch" --graphs --warmup 1 --seed 0 \
        --output-len "$output_len" --stop-at-eos --out "$out/$name.csv" "$@" > /dev/null
}

guesses() {
    case "$1" in
    k0) echo "--speculate 0" ;;
    k8) echo "--speculate $draft" ;;
    k8a) echo "--speculate $draft --speculate-adaptive" ;;
    esac
}

names() {
    python3 -c 'import json, sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$1"
}

one_prompt() {
    python3 -c 'import json, sys; p = [p for p in json.load(open(sys.argv[1]))["prompts"] if p["name"] == sys.argv[2]]; json.dump({"prompts": p}, open(sys.argv[3], "w"))' "$1" "$2" "$3"
}

for r in $(seq 1 "$repeats"); do
    for set in repeat chat; do
        for k in k0 k8 k8a; do
            for n in $seqs; do
                # shellcheck disable=SC2046
                bench "running_${set}_n${n}_${k}_r$r" --prompts "$sets/$set.json" --num-requests "$n" $(guesses $k)
            done
        done
        for name in $(names "$sets/$set.json"); do
            one_prompt "$sets/$set.json" "$name" "$sets/one.json"
            for k in k0 k8 k8a; do
                # shellcheck disable=SC2046
                bench "sampled_${set}_${name}_${k}_r$r" --prompts "$sets/one.json" --num-requests 1 \
                    --temperature "$temperature" --top-p "$top_p" $(guesses $k)
            done
            if [ -d "$draft_model" ]; then
                for d in $draft_lengths; do
                    bench "draft_${set}_${name}_d${d}_r$r" --prompts "$sets/one.json" --num-requests 1 \
                        --speculate "$d" --draft-model "$draft_model"
                done
            fi
        done
    done
    for rate in $rates; do
        for k in k0 k8 k8a; do
            # shellcheck disable=SC2046
            bench "load_chat_q${rate}_${k}_r$r" --prompts "$sets/chat.json" --num-requests "$load_requests" \
                --request-rate "$rate" $(guesses $k)
        done
    done
done
