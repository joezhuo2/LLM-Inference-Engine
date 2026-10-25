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

llama.cpp is not a dependency of this repo. `scripts/setup_llamacpp.sh` clones the pinned release (`b11870`, commit 74bca3950) into `~/.cache/llmie/llama.cpp`, builds only `llama-bench` and `llama-batched-bench` with CUDA for sm_120 (`LLAMA_CUDA_ARCH=89` for an L4 or a 4090; the build takes about 15 minutes on the laptop) and converts the model to `models/tinyllama-bf16.gguf` with llama.cpp's own `convert_hf_to_gguf.py --outtype bf16` (the BF16 weights unchanged, 2.2 GB). It reuses what already exists, so rerunning it is cheap.

`scripts/baseline_llamacpp.py` has two modes. `batched` is the one `scripts/reproduce.sh` uses, because it gives the whole curve with one method:

```bash
~/.venvs/llmie/bin/python scripts/baseline_llamacpp.py batched --model models/tinyllama-bf16.gguf \
    --llama-batched-bench ~/.cache/llmie/llama.cpp/build/bin/llama-batched-bench \
    --prompt-len 64 --output-len 64 --max-batch 1,2,4,8,16 --repetitions 3 --out-dir results
```

It runs `llama-batched-bench -npp <prompt-len> -ntg <output-len - 1> -npl <sizes> --output-format jsonl -ngl 99` once per repetition: for each size B, B sequences with their own prompts are prefilled in one call and then decoded together, one token per sequence per step. That is a static batch of B requests that all arrive at once, the same shape as the HF baseline's static batches. Each row becomes a summary (`driver` `llamacpp_batched`, `results/llamacpp_b<B>_r<run>_summary.csv`) with B requests whose first token comes at `t_pp` (the prefill returns the first token's logits, hence `output-len - 1` decode steps) and which all finish at `t`, so TPOT is `t_tg / (output-len - 1)` and output throughput is `B * output-len / t`. The first configuration of every llama-batched-bench run pays a cold start (prefill 0.49 s instead of 0.018 s at batch 1 on the laptop), so the script asks for the first size twice and drops the first row. Options after `--` replace `-ngl 99`; `--from-jsonl` parses saved outputs, one file per repetition.

`bench` gives the conventional batch-1 numbers from `llama-bench`:

```bash
~/.venvs/llmie/bin/python scripts/baseline_llamacpp.py bench --model models/tinyllama-bf16.gguf \
    --llama-bench ~/.cache/llmie/llama.cpp/build/bin/llama-bench --prompt-len 64 --output-len 64 \
    --summary results/llamacpp_bench.csv
```

`llama-bench` measures prompt processing and generation separately and has no request concurrency, so this is the batch-1 point only (`driver` `llamacpp_bench`), with TTFT derived as `prompt_len / prefill tokens per second` and TPOT as `1 / decode tokens per second`; `--from-json` parses a saved `llama-bench -o json`. Throughput is computed from each repetition's `samples_ns` rather than the printed `samples_ts`: on the laptop (WSL2) one repetition in about twenty reported a duration that wrapped below zero (18446744071733648849 ns, printed as 3.5e-9 tokens/s), and such samples are dropped with a note on stderr. The first prompt repetition is about twice as slow as the rest even after llama-bench's warm-up, which the p50 absorbs. It is a cross-check, not part of reproduce.sh (a second batch-1 series would be merged into the first by the plots): on 2026-10-24 on the laptop it measured 117 decode tokens/s (8.5 ms TPOT), against 8.7 ms at batch 1 from `batched`.

Comparability: the same BF16 weights (not a quantized GGUF), the same prompt and output length, the same GPU, plugged in, several repetitions. The llama.cpp release is stored in the summary's `version` and the flash attention setting in `build` (`fa=auto`, which is on for CUDA). What differs: llama.cpp keeps its KV cache in F16 (the engine in BF16, same size), feeds random tokens instead of the workload's prompts (the timing does not depend on the token values) and does not sample (it computes the logits of every step but never runs a sampler, which the engine does on the GPU).

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
