# M2 plan: correct naive engine (due Oct 18)

Exit criteria (from `build_plan.md`): batch-1 greedy generation of TinyLlama produces readable text; per-layer hidden states match HF within tolerance; teacher-forced argmax match is 99%+ (with mean abs logit error under 0.05, and free-running greedy matching HF for the first 64 tokens on at least 18 of the 20 prompts).

## Decisions Joe confirmed on Oct 9 ("yes to all")

1. SentencePiece: installed copy first, otherwise fetched at build time, like the other dependencies. Implemented as pkg-config first (Ubuntu and Homebrew ship `.pc` files, not CMake packages), then FetchContent v0.2.1. CI: apt `libsentencepiece-dev` on Ubuntu and in the CUDA container; macOS fetches.
2. Claude drafts every naive kernel (Part B: embedding, RMSNorm, RoPE, SwiGLU, residual add, attention over a contiguous KV cache, which also covers causal prefill); Joe explains every line. Joe did not take any of them himself.
3. Order: `dump_reference.py`, then the kernels in forward-pass order, each against a CPU reference, then the forward pass with greedy argmax, then layer 0 against HF, all 22 layers, then logits.
4. Golden format: safetensors per prompt plus a JSON manifest (see below).
5. Scope: batch 1, contiguous KV cache, greedy, tokenizer and chat template, `engine generate`. Temperature sampling waits until the teacher-forced match passes (optional for M2).

## Branches

| # | Branch | Part | Content | State |
| --- | --- | --- | --- | --- |
| 10 | tokenizer/sentencepiece | A | SentencePiece dependency, `Tokenizer`, `chat_prompt`, tokenizer golden, `prompts.json`, `dump_reference.py`, reference self-check | Merged (#19) |
| 11 | kernel/naive-oracles | B | Naive embedding, RMSNorm, RoPE, residual add, SwiGLU with CPU-reference GPU tests and traced-layer checks | Merged (#20) |
| 12 | kernel/naive-attention | B | Causal attention over a contiguous KV cache (GQA, FP32 softmax), CPU reference and `layers.{i}.attn` check on all 22 layers | Merged (#21) |
| 13 | model/forward | B (wiring) and A (harness, CLI) | Batch-1 forward pass, contiguous KV cache, greedy argmax, golden harness, `engine generate` | Merged (#22) |
| 14 | sampling/temperature | B (kernel, generate) and A (CLI) | Gumbel-max temperature sampling, `SamplingParams`, `--temperature`/`--seed` | Merged (#23); M2 closed Oct 9 |

## Reference format (`tests/golden/reference/`, generated, gitignored)

- `manifest.json`: `transformers`, `torch`, `device`, `dtype` (bfloat16), `attn_implementation` (eager), `max_new_tokens` (64), descriptions of `tensors` and `trace_tensors`, and `prompts`: per prompt `name`, `file`, `messages`, `prompt_len`, `total_len`, `generated_ids`, `generated_text`, `trace`.
- `<name>.safetensors` per prompt: `input_ids` I32 `[T]` (chat-template prompt, no BOS, then HF's greedy continuation up to 64 tokens, stopping after EOS); BF16 `embed`, `layers.{i}.out` (i = 0..21), `final_norm`, all `[T, 2048]`; BF16 `logits` `[T, 32000]` (row t predicts token t + 1). One traced prompt (default the first, `kv_cache`) also has per layer `ln1`, `q`, `k`, `v` (pre-RoPE), `attn` (o_proj input), `o`, `ln2`, `gate`, `up`, `act` (down_proj input, silu(gate) * up), `down`.
- C++ access: `tests/support/reference.h` (`reference_dir()`, `load_reference_manifest()`); tensors are read with `MappedFile` and `parse_safetensors_header`.
- HF's full-pass argmax agrees with its own KV-cached greedy tokens at 1,145 of 1,151 positions (99.48%); the teacher-forced metric must compare against the stored logits, not `generated_ids`. The free-running criterion compares against `generated_ids`.

## Results (branch 13, Oct 9)

- Teacher-forced argmax 99.57% (target 99%), mean abs logit error 0.0285 (target under 0.05), readable text from `engine generate`.
- Per-layer tolerance (the open question below): relative L2 per layer, measured worst 2.6%, bound 5% (layer 0: 0.35%, bound 1%).
- Free-running: 15 of 20 identical; the 18-of-20 bar is unreachable in BF16 (HF agrees with itself on 14 of 20). Replaced by: identical, or first divergence at most 2 BF16 ulps apart in HF's logits, on all 20 prompts (Joe confirmed on Oct 9, option 1, no floor on the identical count).
- Streaming: yes, through `TextStream` (decode everything, emit the new suffix, hold back a trailing U+FFFD).

## Open questions for later in M2

- Per-layer tolerance for "hidden states match HF": propose max abs error relative to the layer's RMS, measured once the forward pass exists, rather than guessing now.
- Whether `engine generate` should stream tokens as they decode (needs incremental detokenization: decode the whole sequence each step and print the new suffix).
