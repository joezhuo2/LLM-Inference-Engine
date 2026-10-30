# Branch 44 `tools/llamacpp-baseline` (M5, Part A)

Local in the worktree, on eab6655, not pushed. Every commit gated with the GPU (debug 571, cpu 192).

1. 83d51d1 `scripts/setup_llamacpp.sh`: clones llama.cpp into `~/.cache/llmie/llama.cpp`, pins release b11870 (74bca3950), builds only llama-bench and llama-batched-bench with CUDA (`LLAMA_CUDA_ARCH`, default 120; 89 for L4/4090), converts `models/tinyllama` to `models/tinyllama-bf16.gguf` (llama.cpp's convert_hf_to_gguf.py, its own gguf-py, no new venv deps). About 15 minutes on the laptop.
2. 784b219 llama-bench throughput from `samples_ns`, wrapped (>= 2^63) durations dropped. Found on the real binary: one repetition of 23 reported 18446744071733648849 ns (WSL2 clock).
3. 6dd9ace `baseline_llamacpp.py` gets subcommands `bench` and `batched` (llama-batched-bench jsonl, first size run twice and the cold-start row dropped, version from `--version`, GPU from nvidia-smi, `-pps` rejected); reproduce.sh runs `batched` at 1/2/4/8/16 by default when the setup script's binary and GGUF exist (`--skip-llamacpp`); old `LLAMA_BENCH` path removed.
4. eafb664 `--num-requests N`: one measured batch replayed as N/B consecutive static batches (HF rule), so TTFT includes queueing like the engine's. Without it TTFT read 23 ms vs the engine's 7,800 ms.
5. Plot end labels spread apart (`spread_labels`) since llama.cpp and eager engine end at 1284/1264.
6. Results on eafb664 (2026-10-25): graphs vs llama.cpp 127.7 vs 110.1 tok/s at batch 1 (7.8 vs 9.0 ms TPOT), 479 vs 396 at 4, 1402 vs 1284 at 16; eager engine level with llama.cpp from batch 8. llama-bench cross-check: 117 tok/s decode at batch 1.

Notes for Joe: llama.cpp KV cache is F16, it feeds random tokens and never samples; fa=auto. The batched rows assume N is a multiple of B. changelog and learning log sections are "tools/llamacpp-baseline" / "llama.cpp Baseline (Branch 44)"; branch 43's sections will conflict at the same spots (keep both).
