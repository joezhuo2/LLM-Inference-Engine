# Baselines and plots

The engine's numbers mean little alone. This is how the same workload is run through HuggingFace transformers and llama.cpp and how the results are plotted. Every system writes the engine's summary CSV columns, so one script plots them together.

## The workload

`engine bench` builds its requests from `--seed`, `--prompt-len`, `--output-len`, `--request-rate` and `--num-requests` with its own SplitMix64 (`src/bench/workload.cpp`). `scripts/bench_workload.py` is the same generator in Python and a test pins the same tokens, so a baseline given the same options serves byte-identical prompts and output lengths. Requests ignore EOS and always produce their full output length, with greedy sampling, in every system.

## HuggingFace transformers

```bash
~/.venvs/llmie/bin/python scripts/baseline_hf.py --model models/tinyllama \
    --max-batch 1 --num-requests 32 --prompt-len 64 --output-len 64 --out results/hf_b1.csv
```

BF16 weights, `sdpa` attention (`--attn eager` to compare), a hand-written greedy loop over the KV cache. `--max-batch 1` serves one request at a time; `--max-batch N` serves static batches: a batch starts when its last request has arrived, runs until its longest request is done, and each request is timed to its own last token (which flatters the baseline slightly). The summary's `driver` is `hf_sequential` or `hf_static_batch`.

## llama.cpp

llama.cpp is not a dependency of this repo. `scripts/setup_llamacpp.sh` clones the pinned release (`b11870`, commit 74bca3950) into `~/.cache/llmie/llama.cpp`, builds only `llama-bench` and `llama-batched-bench` with CUDA for sm_120 (`LLAMA_CUDA_ARCH=89` for an L4 or a 4090; the build takes about 15 minutes on the laptop) and converts the model to `models/tinyllama-bf16.gguf` with llama.cpp's own `convert_hf_to_gguf.py --outtype bf16` (the BF16 weights unchanged, 2.2 GB). It reuses what already exists, so rerunning it is cheap. Then:

```bash
llama-bench -m tinyllama-bf16.gguf -p 128 -n 256 -r 5 -ngl 99 -o json > llama.json
python3 scripts/baseline_llamacpp.py --from-json llama.json --prompt-len 128 --output-len 256 \
    --summary results/llamacpp_b1_summary.csv
```

or let the script run it: `--model tinyllama-bf16.gguf --llama-bench /path/to/llama-bench`. `llama-bench` measures prompt processing and generation separately and has no request concurrency, so this gives the batch-1 point only (`driver` `llamacpp_bench`), with TTFT derived as `prompt_len / prefill tokens per second` and TPOT as `1 / decode tokens per second`. For a concurrency curve use `llama-batched-bench`; its output is not parsed yet. The script was written against the documented JSON fields and tested on fixtures, not against a real binary.

Comparability: the same BF16 weights (not a quantized GGUF), the same prompt and output length, the same GPU, plugged in, several repetitions. Record the llama.cpp commit; it is stored in the summary's `version`.

## Plots

```bash
~/.venvs/llmie/bin/python scripts/plot.py results/*_summary.csv --out-dir results/plots
```

Writes `throughput_vs_batch.png`, `tpot_vs_batch.png`, `batch1_bandwidth.png` (batch-1 weight bandwidth against the measured peak and the 60% target) and `summary_table.md`, the same numbers as a table. Repeated runs of one system and batch size are aggregated to their median. `--theme dark` for dark-background pages, `--peak-gb-s` for another GPU. Engine summaries from the batched driver do not carry `bandwidth_gb_s`; at batch 1 it is computed from `tpot_s_p50` and the 2.07 GB of weights.

## Reading the comparison fairly

- HF here is an eager PyTorch loop: at batch 1 it is limited by Python and kernel-launch overhead, not by memory bandwidth, which is the expected result and not a statement about PyTorch kernels.
- Static batching waits for the longest request of a batch; the engine batches continuously. The workload with equal output lengths hides that difference, a workload with `--output-len MIN:MAX` shows it.
- TTFT in an offline batch (every request arrives at once) includes waiting for earlier requests, in every system. Compare TTFT under `--request-rate`.
- The laptop GPU's clocks vary between runs by 10 to 20%; compare medians of several runs, interleaved.
