# Branch 45 tools/roofline

5 commits on main 1ad25fa, not pushed:

1. d98bce3 `scripts/gemm_peak.cu`: square BF16 cuBLAS GEMMs (2048/4096/8192), prints `peak_tflops=`. Laptop: 34.6 / 31.5 / 32.0 TFLOP/s; docs/hardware.md records 34.5 and the ridge point 107 FLOP/byte.
2. 0a6a92c `scripts/roofline.py`: FLOPs and bytes per GEMM and attention from config.json; op times from profile_report.py op tables (`--profile file:rows`); whole step from median TPOT. Writes roofline.png and roofline.md. Elementwise ops left out (about 3% of kernel time). 13 tests.
3. 7d60861 reproduce.sh: batch-16 nsys profile (`results/decode_step_b16.md`), roofline run, `PEAK_TFLOPS` (default 34.5) in environment.txt; make_results fills `auto:roofline`; results.md Roofline section.
4. f0396c2 attention labels on the right of their points.
5. b4b79bc results from reproduce.sh --skip-tests on 7d60861.

Decisions for Joe to check: compute ceiling is measured (cuBLAS), not the datasheet, same as the bandwidth ceiling; the bytes model counts the KV cache once per row (no cache effects), so attention's % of roof is relative to that ideal; whole-step points use TPOT, which includes host time and interleaved prefills.

Finding worth a Part C branch: at batch 16 attention is 19% of kernel time at 7% of its roof.
