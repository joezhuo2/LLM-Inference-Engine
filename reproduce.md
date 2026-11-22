# Reproduce

How to run the engine and how to measure everything [results.md](results.md) reports again. Build first: see [Building](README.md#building).

## Running

```bash
./build/release/bin/engine generate --model models/tinyllama --prompt "Write a haiku about GPUs."
```

The answer streams to stdout as it decodes, followed on stderr by the prompt and generated token counts, the time to the first token and the decode rate. `--system` adds a system message and `--max-new-tokens` (default 256) caps the answer. Decoding is greedy by default; `--temperature 0.7` samples from `softmax(logits / 0.7)` on the GPU with the Gumbel-max trick, `--top-k K` and `--top-p P` then restrict each draw to the K most likely tokens or to the most likely tokens whose probability reaches P (applied in that order after the temperature, as in Hugging Face), and `--seed N` makes the answer repeatable (without it the seed is random and printed on stderr). `generate` serves one request through the contiguous `ModelRunner`, the naive-kernel engine the correctness tests use as their oracle, not through the paged runner and fast kernels that `bench --max-batch` serves: 125 to 135 tokens/s on the RTX 5060, greedy or sampled, with or without top-k and top-p. `--speculate N` (with `--speculate-adaptive` and `--draft-model DIR`, as on `bench`) decodes speculatively: the request then runs through the scheduler and the paged cache with N + 1 rows per step, and stderr also gives the steps and the draft tokens kept. `--tensor-parallel N` cuts the weights into N shards and runs them all on this GPU through `ShardedRunner`, summing their partial results twice per layer; it checks the split and cannot be faster than one model (`results.md`, "Tensor parallelism: a forward pass on the shards").

```bash
./build/release/bin/engine bench --model models/tinyllama --num-requests 64 --prompt-len 128 --output-len 256 --out results/laptop_b1.csv
```

`bench` serves a synthetic workload and writes two CSVs: one row per request (`--out`: prompt and output tokens, preemptions, arrival time, TTFT, TPOT and latency in seconds) and one summary row (`results/laptop_b1_summary.csv`, or `--summary`) with the run's settings, the GPU, the build type, throughput, the mean, p50, p90 and p99 of TTFT, TPOT and latency, and the achieved bandwidth (the 2.07 GB of weights a decode step reads, divided by the median TPOT). Prompts are random token ids after BOS, or with `--prompts scripts/prompts.json` the chat prompts of that file in turn (rendered with the chat template; `--prompt-len` is then ignored); `--prompt-len` and `--output-len` take a length `N` or a range `MIN:MAX` drawn per request, `--seed` fixes the prompts, and `--request-rate R` spreads arrivals as a Poisson process at R requests per second instead of submitting all of them at once. Every request generates its full output length unless `--stop-at-eos` is given, and `--warmup` (default 3) untimed requests run first. `--max-batch N` serves through the paged runner with up to N concurrent sequences and `--graphs` (with `--max-batch`) replays CUDA Graphs for the decode steps, with `--graph-steps` (a power of two, default 2) graph sizes per power of two of batch size; the summary records it as `graph_steps` (0 without graphs). `--token-budget N` (with `--max-batch`, at least the max batch) turns on chunked prefill: a step holds at most N tokens, every decoding sequence gets its row first and the rest goes to prompts, which split across steps when they do not fit; 0, the default, keeps whole-prompt prefill steps, and the summary records the value as `token_budget`. `--speculate N` (with `--max-batch`) turns on speculative decoding by prompt lookup with up to N guessed tokens per sequence and step; the summary records it as `speculate`, with the guessed tokens of the run in `drafted` and the ones the model agreed with in `draft_accepted`; `--speculate-adaptive` lets each sequence halve and double its own limit, and `--draft-model DIR` takes the guesses from a smaller model with the same vocabulary. With `--max-batch` the scheduler caches prefixes: a request whose prompt starts with whole 16-token blocks whose keys and values an earlier request has written, and which are still in the KV cache, reuses those blocks and prefills only the rest; `--no-prefix-cache` turns it off, and the summary records it as `prefix_cache`, with the tokens of every admitted sequence in `prefill_tokens`, the ones found in the cache in `cached_tokens` and the ratio in `prefix_hit_rate`. Prompt rows go through `flash_attention_prefill`; `--prefill-attention paged` (with `--max-batch`) keeps the per-row decode kernels, and the summary records the choice as `prefill_attention`. `--trace FILE` (with `--max-batch`) writes every step of the timed run as JSON in the format of `docs/api.md`, and `--profile` brackets the timed run with `cudaProfilerStart` and `cudaProfilerStop` for Nsight (`profiling.md`). Sampling takes the same `--temperature`, `--top-k` and `--top-p` as `generate`, with seed `seed + i` for request `i`. Without `--max-batch` requests run one at a time through the contiguous `ModelRunner`. With every request submitted at once, the TTFT includes the time a request waits behind the earlier ones, which is why the batch-1 TTFT in the results is several seconds. Benchmark only the `release` preset (the command warns otherwise), plugged in and in maximum performance mode. The laptop's clocks move by 10% to 20% between runs, which is why every summary records its date and settings and `scripts/reproduce.sh` reports the median of several runs.

```bash
./build/release/bin/engine serve --model models/tinyllama --graphs
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":64,"stream":true}'
```

`serve` loads the model and answers on `127.0.0.1:8000` (`--host`, `--port`): an OpenAI-compatible `POST /v1/chat/completions`, streamed or not, `GET /health`, `GET /metrics`, and `GET /v1/steps`, a Server-Sent Events stream of every scheduler step in the format of the trace files. One engine thread runs the scheduler while HTTP threads add requests between steps. `--max-batch` (default 16), `--graphs`, `--graph-steps`, `--token-budget`, `--speculate`, `--speculate-adaptive` and `--no-prefix-cache` are those of `bench` (`serve` has no `--draft-model`); `--temperature`, `--top-k`, `--top-p` and `--max-tokens` are the defaults for requests that do not set them; `--cors-origin`, repeatable, lists the browser origins allowed to call it (default `http://localhost:3000` and the deployed site), which is what the site's dashboard needs in live mode; `--allow-host`, repeatable, adds names to the `Host` check below. `docs/api.md` is the contract.

**Security:** the server has no authentication and no rate limit. It binds to loopback by default for that reason: never expose it publicly, forward its port or tunnel it, since anyone who can reach it can use the GPU and read every prompt's answers through `/v1/steps`. A client that disconnects cancels its request, and its blocks come back before the next step. Requests whose `Host` header is not a loopback name, the `--host` address or an `--allow-host` name get 403, which stops DNS rebinding.

## Reproducing the results

From a clean clone, after the build above. The Python scripts (golden data, the HuggingFace baseline, plots) need the pinned packages in a virtual environment, which `scripts/reproduce.sh` looks for at `~/.venvs/llmie` (or set `PYTHON`):

```bash
python3 -m venv ~/.venvs/llmie && ~/.venvs/llmie/bin/pip install -r scripts/requirements.txt
```

The golden data the correctness tests compare against is generated from the model, not committed (`tests/golden/` is gitignored):

```bash
~/.venvs/llmie/bin/python scripts/checksum.py --model models/tinyllama --out tests/golden/checksums.json
```

```bash
~/.venvs/llmie/bin/python scripts/tokenizer_golden.py --model models/tinyllama --out tests/golden/tokenizer.json
```

```bash
~/.venvs/llmie/bin/python scripts/dump_reference.py --model models/tinyllama --out tests/golden/reference
```

The llama.cpp baseline is a pinned release (b11870) built outside the repo, plus a BF16 GGUF of the same weights; without it `reproduce.sh` skips the llama.cpp rows. On a GPU other than the RTX 5060 set `LLAMA_CUDA_ARCH` (89 for an L4 or an RTX 4090):

```bash
scripts/setup_llamacpp.sh
```

The vLLM baseline of the load runs is a pinned release (0.32.0) in its own virtual environment, `~/.venvs/llmie-vllm`; without it `reproduce.sh` skips the vLLM rows:

```bash
scripts/setup_vllm.sh
```

Then one script rebuilds the release preset, runs the tests and the HF comparison, benchmarks the engine (eager at max batch 1, 2, 4, 8, 16 and with CUDA Graphs at 1, 4, 16), HuggingFace and llama.cpp, runs vLLM's server and the engine under Poisson arrivals of 1 to 8 requests per second (whole-prompt prefill and token budgets of 256, 512 and 2,048) with the long-prompt stall probe, measures the time to the first token against prompt length, profiles batch-1 and batch-16 decode steps with Nsight Systems when `nsys` is installed, runs the on-chip memory microbenchmarks, draws the plots and the roofline, and rewrites the measured blocks of `results.md`. Plug the laptop in, use maximum performance mode and close other GPU programs first. A full run takes about 10 minutes; `--quick` is a smoke run, and `--skip-tests`, `--skip-hf`, `--skip-llamacpp`, `--skip-vllm`, `--skip-load`, `--skip-prefill` and `--skip-onchip` leave those parts out:

```bash
scripts/reproduce.sh
```

The "% of peak" figures and the roofline use measured ceilings, not datasheet ones: 323.6 GB/s and 34.5 TFLOP/s on the laptop (`hardware.md`). On another GPU measure both and pass them as `PEAK_GB_S` and `PEAK_TFLOPS`:

```bash
nvcc -O3 -arch=native src/practice/bandwith_test.cu -o /tmp/bandwidth_test && /tmp/bandwidth_test
```

```bash
nvcc -std=c++20 -arch=native scripts/gemm_peak.cu -lcublas -o /tmp/gemm_peak && /tmp/gemm_peak
```

The `results/` files and `results.md` are committed right after the code they measured, and `results/environment.txt` records the commit, GPU, driver, CUDA version and date of the run.
