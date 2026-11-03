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
    --prompt-len 64 --output-len 64 --max-batch 1,2,4,8,16 --num-requests 32 --repetitions 3 --out-dir results
```

It runs `llama-batched-bench -npp <prompt-len> -ntg <output-len - 1> -npl <sizes> --output-format jsonl -ngl 99` once per repetition: for each size B, B sequences with their own prompts are prefilled in one call and then decoded together, one token per sequence per step. That is a static batch of B requests that all arrive at once, the same shape as the HF baseline's static batches. Each row becomes a summary (`driver` `llamacpp_batched`, `results/llamacpp_b<B>_r<run>_summary.csv`) in which `--num-requests` N requests, all arriving at once, are served as N / B consecutive batches of that timing: request i runs in batch `k = i // B`, gets its first token at `k * t + t_pp` (the prefill returns the first token's logits, hence `output-len - 1` decode steps) and finishes at `(k + 1) * t`. TPOT is `t_tg / (output-len - 1)` and output throughput `B * output-len / t` whatever N is, but TTFT and latency then include waiting for earlier batches, as they do for the engine and the HF baseline on the same 32-request workload; without `--num-requests` a single batch is reported and TTFT is only the prefill. N should be a multiple of every B (a partial last batch is timed like a full one). The first configuration of every llama-batched-bench run pays a cold start (prefill 0.49 s instead of 0.018 s at batch 1 on the laptop), so the script asks for the first size twice and drops the first row. Options after `--` replace `-ngl 99`; `--from-jsonl` parses saved outputs, one file per repetition.

`bench` gives the conventional batch-1 numbers from `llama-bench`:

```bash
~/.venvs/llmie/bin/python scripts/baseline_llamacpp.py bench --model models/tinyllama-bf16.gguf \
    --llama-bench ~/.cache/llmie/llama.cpp/build/bin/llama-bench --prompt-len 64 --output-len 64 \
    --summary results/llamacpp_bench.csv
```

`llama-bench` measures prompt processing and generation separately and has no request concurrency, so this is the batch-1 point only (`driver` `llamacpp_bench`), with TTFT derived as `prompt_len / prefill tokens per second` and TPOT as `1 / decode tokens per second`; `--from-json` parses a saved `llama-bench -o json`. Throughput is computed from each repetition's `samples_ns` rather than the printed `samples_ts`: on the laptop (WSL2) one repetition in about twenty reported a duration that wrapped below zero (18446744071733648849 ns, printed as 3.5e-9 tokens/s), and such samples are dropped with a note on stderr. The first prompt repetition is about twice as slow as the rest even after llama-bench's warm-up, which the p50 absorbs. It is a cross-check, not part of reproduce.sh (a second batch-1 series would be merged into the first by the plots): on 2026-10-24 on the laptop it measured 117 decode tokens/s (8.5 ms TPOT), against 8.7 ms at batch 1 from `batched`.

Comparability: the same BF16 weights (not a quantized GGUF), the same prompt and output length, the same GPU, plugged in, several repetitions. The llama.cpp release is stored in the summary's `version` and the flash attention setting in `build` (`fa=auto`, the default). What differs: llama.cpp keeps its KV cache in F16 (the engine in BF16, same size), feeds random tokens instead of the workload's prompts (the timing does not depend on the token values) and does not sample (it computes the logits of every step but never runs a sampler, which the engine does on the GPU).

## vLLM

HuggingFace and llama.cpp run static batches, so neither can serve requests that arrive at their own times. vLLM is the online comparison. `scripts/setup_vllm.sh` installs the pinned release (0.32.0, with torch 2.13.0) into its own environment, `~/.venvs/llmie-vllm`, because vLLM pins its own torch. `scripts/baseline_vllm.py` then starts the server, serves each rate and stops it:

```bash
~/.venvs/llmie-vllm/bin/python scripts/baseline_vllm.py --model models/tinyllama --rates 1,2,3,4,5,6,8 --run 1
```

The server runs as `vllm serve models/tinyllama --dtype bfloat16 --max-num-seqs 16 --gpu-memory-utilization 0.7 --max-model-len 2048 --generation-config vllm`. The client sends the workload's token ids to `/v1/completions` at the workload's arrival times (120 requests, prompts of 16 to 512 tokens, outputs of 16 to 256, seed 0, 3 warmup requests per rate), greedy with `ignore_eos`. It writes `results/load/vllm_q<rate>_r<run>.csv` and its summary with the driver `vllm_online`. TTFT counts from the scheduled arrival, and tokens are counted from the `usage` field of each stream chunk instead of from the chunks.

Comparability: the same BF16 weights (the script stops if the server's log reports another dtype), the same token ids, arrival times and output lengths, the same GPU, and the same limit of 16 concurrent sequences as `engine bench --max-batch 16`. What differs:

- **HTTP and Python:** every vLLM token goes through a Python server, HTTP and the Python client, and its times are taken in the client. `engine bench` has no HTTP and times requests inside the process, so this overhead counts only against vLLM.
- **Attention:** vLLM picks its `FLASH_ATTN` backend (FlashAttention 2) here, not FlashInfer, which it uses only for sampling. Decode runs from its own `torch.compile` and CUDA Graph capture, which added about 80 s to the first start.
- **Prefill and decode:** vLLM's default chunked prefill mixes prefill and decode in one step with a budget of 2,048 tokens per step on this GPU (its server default), while the engine's steps are all prefill or all decode.
- **Sampling:** vLLM samples on its own path in a Python step loop; the engine samples on the GPU with its own kernel.
- **Memory:** neither system is short of KV space (0 preemptions in vLLM's runs): vLLM's cache holds 153,888 tokens at 70% of the VRAM.
- **The limit of 16:** this matches the engine's run but is not vLLM's default for this GPU (128 sequences for the server), so these curves do not show what vLLM does with more concurrency. A sweep of both systems past 16 would be the fairer test at high rates.

Reading (RTX 5060 Laptop GPU, 2026-11-03, median of 3 runs, 0 preemptions, every stream chunk carried exactly one token):

| QPS | output tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 167 | 46 | 103 | 8.9 | 10.5 |
| 2 | 329 | 29 | 39 | 8.8 | 10.4 |
| 3 | 486 | 32 | 45 | 9.1 | 11.1 |
| 4 | 630 | 33 | 43 | 10.4 | 11.4 |
| 5 | 774 | 35 | 194 | 10.5 | 11.2 |
| 6 | 912 | 36 | 480 | 10.6 | 11.4 |
| 8 | 1,105 | 42 | 1,010 | 10.9 | 11.4 |

The client's own delay in sending a request reached at most 50 ms (p99 at most 34 ms, in the slower rates) and is in these TTFTs, since they count from the scheduled arrival.

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
