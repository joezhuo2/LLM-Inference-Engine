#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/speculative_sweep.sh [--quick]

Runs engine bench --speculate (#102) on one request at a time, so the other rows of the batch are
free for drafts: every prompt of scripts/prompts_repeat.json (answers that repeat their prompt)
and the first CHAT_PROMPTS prompts of scripts/prompts.json (chat answers) alone, greedy, to EOS or
OUTPUT_LEN tokens, once per draft length, and writes results/speculative/<set>_<prompt>_k<K>_r<run>.csv with their
_summary.csv. The first run of every prompt and draft length also writes a trace to TRACES, which
scripts/speculative_report.py uses to say whether the output changed, scripts/speculative_ties.py
to look at the logits where it did, and scripts/speculative_example.py to rebuild the steps of
one answer for the site.

Environment: ENGINE (default build/release/bin/engine), MODEL (default models/tinyllama), DRAFTS
(default "0 2 4 8 15"), REPEATS (3), OUTPUT_LEN (256), MAX_BATCH (16), CHAT_PROMPTS (6),
OUT (default results/speculative), TRACES (default OUT/traces, not committed).
USAGE
    exit 2
}

drafts=${DRAFTS:-0 2 4 8 15}
repeats=${REPEATS:-3}
for arg in "$@"; do
    case "$arg" in
    --quick)
        drafts="0 8"
        repeats=1
        ;;
    *) usage ;;
    esac
done

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
output_len=${OUTPUT_LEN:-256}
max_batch=${MAX_BATCH:-16}
chat_prompts=${CHAT_PROMPTS:-6}
out=${OUT:-results/speculative}
traces=${TRACES:-$out/traces}
root=$(cd "$(dirname "$0")/.." && pwd)
single=$(mktemp -d)
trap 'rm -rf "$single"' EXIT
mkdir -p "$out" "$traces"

names() {
    python3 -c 'import json, sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"][:int(sys.argv[2])]))' "$1" "$2"
}

one_prompt() {
    python3 -c 'import json, sys; p = [p for p in json.load(open(sys.argv[1]))["prompts"] if p["name"] == sys.argv[2]]; json.dump({"prompts": p}, open(sys.argv[3], "w"))' "$1" "$2" "$3"
}

for set in repeat chat; do
    file=$root/scripts/prompts.json
    count=$chat_prompts
    if [ "$set" = repeat ]; then
        file=$root/scripts/prompts_repeat.json
        count=1000
    fi
    for name in $(names "$file" "$count"); do
        one_prompt "$file" "$name" "$single/$name.json"
        for r in $(seq 1 "$repeats"); do
            for k in $drafts; do
                echo "$set $name, up to $k draft tokens, run $r"
                trace=()
                [ "$r" = 1 ] && trace=(--trace "$traces/${set}_${name}_k${k}.json")
                "$engine" bench --model "$model" --max-batch "$max_batch" --graphs --num-requests 1 \
                    --warmup 1 --seed 0 --prompts "$single/$name.json" --output-len "$output_len" \
                    --stop-at-eos --speculate "$k" "${trace[@]}" \
                    --out "$out/${set}_${name}_k${k}_r$r.csv" > /dev/null
            done
        done
    done
done
