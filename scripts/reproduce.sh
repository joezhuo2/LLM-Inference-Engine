#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

usage() {
    cat >&2 <<'USAGE'
usage: scripts/reproduce.sh [--quick] [--skip-tests] [--skip-hf] [--skip-llamacpp]

Rebuilds the release preset and re-measures everything docs/results.md reports:
  results/environment.txt   commit, GPU, driver, CUDA, date, workload
  results/correctness.txt   teacher-forced and free-running comparison with HuggingFace
  results/engine_b*_r*.csv  engine bench at max batch 1 2 4 8 16, REPEATS runs each
  results/engine_graphs_b*_r*.csv   the same with --graphs (CUDA Graph decode), max batch 1 4 16
  results/hf_b*_r*.csv      HuggingFace baseline at max batch 1 4 16, HF_REPEATS runs each
  results/llamacpp_b*_r*_summary.csv   llama.cpp at batch 1 2 4 8 16, REPEATS runs each, when
                            scripts/setup_llamacpp.sh has built it (skipped otherwise)
  results/decode_step.md    nsys launch gaps, GPU time per op and bandwidth of a batch-1 decode step
  results/decode_step_graphs.md   the same decode step replayed from a CUDA Graph
  results/decode_step_b16.md   the GPU time per op of a decode step with 16 rows
  results/plots/roofline.*  the ops and the whole decode step on a roofline (scripts/roofline.py)
  results/plots/            the plots and summary_table.md
and then refreshes the measured blocks of docs/results.md.

Environment: MODEL (default models/tinyllama), PEAK_GB_S (measured DRAM bandwidth, 323.6),
PEAK_TFLOPS (measured BF16 GEMM peak from scripts/gemm_peak.cu, 34.5),
REPEATS (3), HF_REPEATS (1), PYTHON (default ~/.venvs/llmie/bin/python, else python3),
LLAMA_BATCHED_BENCH and LLAMA_GGUF (default where scripts/setup_llamacpp.sh puts them).
Plug the laptop in, use maximum performance mode and close other GPU programs first.
USAGE
    exit 2
}

quick=0
skip_tests=0
skip_hf=0
skip_llamacpp=0
for arg in "$@"; do
    case "$arg" in
    --quick) quick=1 ;;
    --skip-tests) skip_tests=1 ;;
    --skip-hf) skip_hf=1 ;;
    --skip-llamacpp) skip_llamacpp=1 ;;
    *) usage ;;
    esac
done

model=${MODEL:-models/tinyllama}
peak=${PEAK_GB_S:-323.6}
peak_tflops=${PEAK_TFLOPS:-34.5}
repeats=${REPEATS:-3}
hf_repeats=${HF_REPEATS:-1}
python=${PYTHON:-$HOME/.venvs/llmie/bin/python}
llama_batched=${LLAMA_BATCHED_BENCH:-$HOME/.cache/llmie/llama.cpp/build/bin/llama-batched-bench}
llama_gguf=${LLAMA_GGUF:-models/tinyllama-bf16.gguf}
[ -x "$python" ] || python=python3
requests=32
prompt=64
output=64
if [ "$quick" = 1 ]; then
    requests=8
    repeats=1
fi
out=results
mkdir -p "$out/plots"

echo "== environment"
{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    dirty=""
    git diff --quiet HEAD -- . ':!results' ':!docs/results.md' || dirty=" (uncommitted changes)"
    echo "commit=$(git rev-parse --short HEAD)$dirty"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "cuda=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
    echo "build=Release"
    echo "peak_gb_s=$peak"
    echo "peak_tflops=$peak_tflops"
    echo "workload=$requests requests, $prompt prompt tokens, $output output tokens, greedy, seed 0, median of $repeats run(s)"
} > "$out/environment.txt"
cat "$out/environment.txt"

echo "== build"
cmake --preset release > /dev/null
cmake --build --preset release

if [ "$skip_tests" = 0 ]; then
    echo "== tests"
    ctest --preset release -LE todo_joe --output-on-failure
fi

echo "== correctness against HuggingFace"
build/release/bin/gpu_tests --gtest_filter='ForwardGolden.*:GenerateTest.FreeRunning*' | tee "$out/correctness.txt"

quiet() {
    local log
    if ! log=$("$@" 2>&1); then
        echo "$log" >&2
        echo "failed: $*" >&2
        exit 1
    fi
    echo "$log" | grep -E "tokens/s" | head -1 || true
}

bench() {
    quiet "build/release/bin/engine" bench --model "$model" --num-requests "$requests" \
        --prompt-len "$prompt" --output-len "$output" --max-batch "$1" --out "$out/engine_b$1_r$2.csv"
}

echo "== engine bench"
for r in $(seq 1 "$repeats"); do
    for b in 1 2 4 8 16; do
        echo -n "batch $b run $r: "
        bench "$b" "$r"
    done
done

graph_bench() {
    quiet "build/release/bin/engine" bench --model "$model" --num-requests "$requests" \
        --prompt-len "$prompt" --output-len "$output" --max-batch "$1" --graphs \
        --out "$out/engine_graphs_b$1_r$2.csv"
}

echo "== engine bench with CUDA Graphs"
for r in $(seq 1 "$repeats"); do
    for b in 1 4 16; do
        echo -n "graphs batch $b run $r: "
        graph_bench "$b" "$r"
    done
done

if [ "$skip_hf" = 0 ]; then
    echo "== HuggingFace baseline"
    for r in $(seq 1 "$hf_repeats"); do
        for b in 1 4 16; do
            echo -n "hf batch $b run $r: "
            quiet "$python" scripts/baseline_hf.py --model "$model" --max-batch "$b" \
                --num-requests "$requests" --prompt-len "$prompt" --output-len "$output" \
                --out "$out/hf_b${b}_r$r.csv"
        done
    done
fi

if [ "$skip_llamacpp" = 0 ]; then
    if [ -x "$llama_batched" ] && [ -s "$llama_gguf" ]; then
        echo "== llama.cpp baseline"
        rm -f "$out"/llamacpp_*_summary.csv
        "$python" scripts/baseline_llamacpp.py batched --model "$llama_gguf" \
            --llama-batched-bench "$llama_batched" --prompt-len "$prompt" --output-len "$output" \
            --max-batch 1,2,4,8,16 --num-requests "$requests" --repetitions "$repeats" --out-dir "$out"
    else
        echo "== llama.cpp baseline skipped: run scripts/setup_llamacpp.sh first"
    fi
fi

if command -v nsys > /dev/null; then
    echo "== decode step profile"
    MODEL=$model PEAK_GB_S=$peak scripts/profile.sh nsys decode_step --max-batch 1 \
        --num-requests 4 --prompt-len "$prompt" --output-len "$output" > /dev/null
    cp results/profile/decode_step.md "$out/decode_step.md"
    MODEL=$model PEAK_GB_S=$peak scripts/profile.sh nsys decode_step_graphs --max-batch 1 --graphs \
        --num-requests 4 --prompt-len "$prompt" --output-len "$output" > /dev/null
    cp results/profile/decode_step_graphs.md "$out/decode_step_graphs.md"
    MODEL=$model PEAK_GB_S=$peak scripts/profile.sh nsys decode_step_b16 --max-batch 16 \
        --num-requests 16 --prompt-len "$prompt" --output-len "$output" > /dev/null
    cp results/profile/decode_step_b16.md "$out/decode_step_b16.md"
fi

echo "== plots and results page"
"$python" scripts/plot.py "$out"/*_summary.csv --out-dir "$out/plots" --peak-gb-s "$peak"
"$python" scripts/roofline.py "$out"/*_summary.csv --profile "$out/decode_step.md:1" \
    --profile "$out/decode_step_b16.md:16" --config "$model/config.json" \
    --context $((prompt + output / 2)) --out-dir "$out/plots" --peak-gb-s "$peak" \
    --peak-tflops "$peak_tflops"
"$python" scripts/make_results.py --results-dir "$out" --doc docs/results.md --peak-gb-s "$peak"
echo "done: review docs/results.md and results/plots, then commit results/ and the doc together"
