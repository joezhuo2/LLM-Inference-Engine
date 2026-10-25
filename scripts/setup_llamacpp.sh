#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/setup_llamacpp.sh

Builds the pinned llama.cpp release with CUDA (llama-bench and llama-batched-bench only) outside
the repo and converts the model to a BF16 GGUF, the comparison point for scripts/baseline_llamacpp.py
and scripts/reproduce.sh. Safe to rerun: the checkout, the build and an existing GGUF are reused.

Environment: LLAMA_CPP_DIR (default ~/.cache/llmie/llama.cpp), LLAMA_CPP_REF (default b11870),
LLAMA_CUDA_ARCH (default 120, the RTX 5060; 89 for an L4 or a 4090), MODEL (default
models/tinyllama), LLAMA_GGUF (default models/tinyllama-bf16.gguf), PYTHON (default the llmie venv).
USAGE
    exit 2
}

[ $# -eq 0 ] || usage

dir=${LLAMA_CPP_DIR:-$HOME/.cache/llmie/llama.cpp}
ref=${LLAMA_CPP_REF:-b11870}
arch=${LLAMA_CUDA_ARCH:-120}
model=${MODEL:-models/tinyllama}
gguf=${LLAMA_GGUF:-models/tinyllama-bf16.gguf}
python=${PYTHON:-$HOME/.venvs/llmie/bin/python}
[ -x "$python" ] || python=python3

echo "== llama.cpp $ref in $dir"
if [ ! -d "$dir/.git" ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$dir"
fi
git -C "$dir" fetch --tags --quiet origin
git -C "$dir" -c advice.detachedHead=false checkout --quiet "$ref"

echo "== build (CUDA arch $arch)"
cmake -S "$dir" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build "$dir/build" --target llama-bench llama-batched-bench -j "$(nproc)"

echo "== BF16 GGUF"
if [ -s "$gguf" ]; then
    echo "reusing $gguf"
else
    "$python" "$dir/convert_hf_to_gguf.py" "$model" --outtype bf16 --outfile "$gguf"
fi

echo "done: $dir/build/bin/llama-bench, $dir/build/bin/llama-batched-bench, $gguf"
