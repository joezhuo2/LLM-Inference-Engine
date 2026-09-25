#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git ls-files '*.cpp' '*.h' '*.cu' '*.cuh' | xargs -r clang-format -i
