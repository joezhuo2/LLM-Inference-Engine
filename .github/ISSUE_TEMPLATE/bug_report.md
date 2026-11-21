---
name: Bug report
about: Something crashes, gives wrong output or a wrong number
title: ""
labels: bug
---

## What happened

## What you expected

## Steps to reproduce

The exact commands, from a clean build if you can:

```bash

```

## Environment

- Commit (`git rev-parse --short HEAD`):
- GPU and driver (`nvidia-smi --query-gpu=name,driver_version --format=csv`):
- CUDA toolkit (`nvcc --version`):
- OS (and WSL version, if any):
- CMake preset (`release`, `debug` or `cpu`):
- Model (TinyLlama-1.1B-Chat-v1.0 or another):

## Output

The error, the failing test's output or the wrong tokens. For a wrong number, attach the CSV from `results/` and `results/environment.txt`.

```

```

## Checked

- [ ] `ctest --preset release -LE todo_joe` output is attached or the failing test is named
- [ ] For a kernel bug: `compute-sanitizer` output is attached
