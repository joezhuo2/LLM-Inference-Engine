---
name: Result that does not reproduce
about: scripts/reproduce.sh gave numbers far from results.md
title: ""
labels: results
---

## Which number

The table or figure in `results.md` and the value you got.

## How it was run

- [ ] `scripts/reproduce.sh` from a clean clone, as in `reproduce.md`
- Commit:
- GPU, driver and CUDA version (or attach `results/environment.txt`):
- Measured ceilings passed as `PEAK_GB_S` and `PEAK_TFLOPS`:

## Files

Attach the CSVs from `results/` that differ.
