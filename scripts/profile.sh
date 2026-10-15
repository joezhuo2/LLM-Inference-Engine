#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

usage() {
    cat >&2 <<'USAGE'
usage:
  scripts/profile.sh nsys <tag> [engine bench options]
  scripts/profile.sh ncu <tag> <kernel-regex> [engine bench options]

Runs `engine bench --profile` from the release build under Nsight Systems (a timeline with
the NVTX ranges, then scripts/profile_report.py) or Nsight Compute (one report per matching
kernel launch). Output goes to results/profile/<tag>*. Environment: ENGINE (binary),
MODEL (model directory), PEAK_GB_S (measured DRAM bandwidth, default 323.6),
NCU_SET (section set, default full), NCU_COUNT (launches to profile, default 4).
Bench options default to --max-batch 1 --num-requests 4 --prompt-len 32 --output-len 32.
USAGE
    exit 2
}

[ $# -ge 2 ] || usage
mode=$1
tag=$2
shift 2

engine=${ENGINE:-build/release/bin/engine}
model=${MODEL:-models/tinyllama}
peak=${PEAK_GB_S:-323.6}
out=results/profile
mkdir -p "$out"

if [ ! -x "$engine" ]; then
    echo "missing $engine: run cmake --build --preset release" >&2
    exit 1
fi

kernel=""
if [ "$mode" = ncu ]; then
    [ $# -ge 1 ] || usage
    kernel=$1
    shift
fi

bench=(--model "$model" --max-batch 1 --num-requests 4 --prompt-len 32 --output-len 32)
if [ $# -gt 0 ]; then
    bench=(--model "$model" "$@")
fi
bench+=(--profile --out "$out/$tag.csv")

case "$mode" in
nsys)
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true -o "$out/$tag" "$engine" bench "${bench[@]}"
    python3 scripts/profile_report.py "$out/$tag.nsys-rep" --summary "$out/${tag}_summary.csv" \
        --peak-gb-s "$peak" | tee "$out/$tag.md"
    ;;
ncu)
    ncu --profile-from-start off --kernel-name "regex:$kernel" \
        --launch-count "${NCU_COUNT:-4}" --set "${NCU_SET:-full}" \
        --force-overwrite -o "$out/${tag}_ncu" "$engine" bench "${bench[@]}"
    echo "report: $out/${tag}_ncu.ncu-rep (open with ncu-ui or: ncu --import <file> --page details)"
    ;;
*)
    usage
    ;;
esac
