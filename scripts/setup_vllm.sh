#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/setup_vllm.sh

Installs the pinned vLLM release into its own virtual environment, the comparison point for
scripts/baseline_vllm.py and scripts/reproduce.sh. vLLM pins its own torch, so it stays out of the
llmie venv. Safe to rerun: an environment that already has the pinned version is reused.

Environment: VLLM_VENV (default ~/.venvs/llmie-vllm), VLLM_VERSION (default 0.32.0).
USAGE
    exit 2
}

[ $# -eq 0 ] || usage

venv=${VLLM_VENV:-$HOME/.venvs/llmie-vllm}
version=${VLLM_VERSION:-0.32.0}

echo "== vLLM $version in $venv"
if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv "$venv"
fi
if [ "$("$venv/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null)" = "$version" ]; then
    echo "reusing vllm $version"
else
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet "vllm==$version"
fi

echo "done: $venv/bin/vllm"
