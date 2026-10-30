#!/usr/bin/env bash
set -euo pipefail
cd /home/crystalflux/code/llm-inference-engine-claude
log=$(dirname "$0")
if ~/scratch/query 2>/dev/null | grep -q 'compute capability: 12.0'; then
    exclude='todo_joe'; echo "GPU: available, running gpu tests"
else
    exclude='todo_joe|gpu'; echo "GPU: NOT available, skipping gpu-labelled tests"
fi
git ls-files '*.cpp' '*.h' '*.cu' '*.cuh' | xargs clang-format --dry-run --Werror
git ls-files -o --exclude-standard '*.cpp' '*.h' '*.cu' '*.cuh' | xargs -r clang-format --dry-run --Werror
echo "format: ok"
for preset in debug cpu; do
    if cmake --build --preset "$preset" 2>&1 | grep -E 'warning:|error:|FAILED:'; then
        echo "build $preset: FAILED or warned"; exit 1
    fi
    ex=$exclude; [ "$preset" = cpu ] && ex='todo_joe'
    ctest --preset "$preset" -LE "$ex" --output-on-failure > "$log/ctest_$preset.log" 2>&1 || {
        grep -E 'Failed|\*\*\*' "$log/ctest_$preset.log" | head -20; echo "ctest $preset: FAILED"; exit 1; }
    grep 'tests passed' "$log/ctest_$preset.log" | sed "s/^/ctest $preset: /"
done
echo "VERIFY OK"
