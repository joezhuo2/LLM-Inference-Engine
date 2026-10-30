# Technical notes

Decisions, gotchas and verified facts from the sessions so far. Each item was hit or checked for real, not assumed.

## Environment

- Toolchain: CUDA 13.3 (`/usr/local/cuda`), CMake 3.28.3, GCC 13.3, Ninja, clang-format 18.1.3, compute-sanitizer. `pkg-config` is installed. Not installed: clangd, the SentencePiece C++ library (so the build fetches v0.2.1), any clang compiler. No sudo, so nothing can be apt-installed from Claude's process.
- Python: `~/.venvs/llmie` has every package in `scripts/requirements.txt` (and pyyaml). The repo's `.venv` lacks sentencepiece and matplotlib; do not use it.
- Model: `models` is an untracked, gitignored symlink to `/home/crystalflux/models` in each checkout (untracked in #18); `models/tinyllama` has config, `tokenizer.model`, `tokenizer.json`, `model.safetensors` (201 BF16 tensors). Pulling a commit that untracks a file deletes it from other checkouts, so after #18 each checkout recreates the link with `ln -s ~/models models`. `model_dir/` is an older gitignored HF clone used by Joe's practice scripts.
- `~/scratch/query` is Joe's device-query binary; it prints `compute capability: 12.0` when the GPU is reachable and garbage (`1.0`) when it is not.
- GPU access from Claude's process is intermittent: `cudaErrorInsufficientDriver` with `libcuda` correctly resolving to `/usr/lib/wsl/lib`. It is environmental; nothing in the repo causes it.
- The session scratchpad (`/tmp/claude-1000/...`) is wiped between sessions; persistent helper files go in `~/.cache/llmie-claude/`.

## Build and CMake

- `FetchContent_Declare(<name> ... FIND_PACKAGE_ARGS NAMES <pkg>)`: the first argument must match the package name, or the installed config warns about a name mismatch on every configure (fixed for nlohmann_json).
- Ubuntu's `libcli11-dev` ships `CLI11Config.cmake`, so CI finds it without downloading; the target is `CLI11::CLI11`.
- `gtest_discover_tests(... PROPERTIES LABELS "a;b")` loses every label after the first in CMake 3.28: `GoogleTestAddTests.cmake` expands properties as an unquoted list and writes `LABELS a b`, so `b` is read as a property name. Escaping (`\;`, `\\;`) does not survive the generated scripts. Use a single label per test target.
- The test targets: `unit_tests` (no label) and `gpu_tests` (`gpu`); `todo_unit_tests` and `todo_gpu_tests` (`todo_joe`) are created with a Part C stub and deleted when empty (`todo_unit_tests` exists since branch 15 for `test_block_manager.cpp`, `todo_gpu_tests` since branch 16 for `test_write_kv.cpp`). `unit_tests` also runs Joe's `src/practice/` `buffer_test` through ctest (119 debug tests on Oct 9 included 7 `BufferTest.*`). `tests/support/` is header-only and on the include path of all of them; `src/` is on the include path of the GPU targets (for `kernels/cuda_check.h`).
- Test path macros: `ENGINE_MODEL_DIR` is a compile definition (default `${PROJECT_SOURCE_DIR}/models/tinyllama`), overridable at run time with the environment variable of the same name. Tests that need the model call `GTEST_SKIP()` when it is missing (CI has no model).

- `add_compile_options` sets a directory property that initializes targets created afterwards, including FetchContent subprojects, so the warning flags are added after the dependency block; otherwise third-party code compiles with `-Wall -Wextra -Wpedantic` and can fail the zero-warning gate. The fetched SentencePiece also gets `-w`, because its own code trips GCC 13's `-Wstringop-overflow` with its own flags.
- `pkg_check_modules` fails if any `Requires.private` module is missing: SentencePiece's `.pc` lists `protobuf-lite` even when built with its internal protobuf, so a hand-built copy is silently skipped (the build falls back to fetching). On Ubuntu `libsentencepiece-dev` pulls in `libprotobuf-dev`, so the apt path works.
- `FetchContent_Declare(... EXCLUDE_FROM_ALL)` (CMake 3.28+) builds only the fetched targets something links, so SentencePiece's `spm_*` tools are never compiled.
- The gate script greps the build output for `warning:`, `error:` and `FAILED:` (with the colons): the bare word `error` matched SentencePiece's `error.cc.o`.

## Compilers

- GCC accepts `EXPECT_THROW(T(f::g()), E)` but Apple Clang parses `T(f::g())` as a declaration (most vexing parse); use `T{...}`.
- gtest assertion macros contain an `if`/`else`, so an unbraced `if (...) EXPECT_...` triggers `-Wdangling-else`; always brace.
- GCC 13's `-Wdangling-reference` fires on `const auto& x = f(temporary).member` (a real bug) and also on `const T& r = helper(local, "temporary string")` (a false positive); return by value in the helper.
- Release builds define `NDEBUG` for CUDA and C++, which is how debug-only code (`KERNEL_CHECK` sync, the NaN fill) is switched.

## CI

- Jobs normally take one to two minutes. On Oct 7 the clang-format and Ubuntu CPU jobs hung for over an hour in `apt-get` because a package mirror for GitHub's Ubuntu runners was unreachable (the CUDA job's container uses a different mirror and was fine). Fixed by `timeout-minutes` (15, or 30 for CUDA), `APT_OPTS` (3 retries, 30-second HTTP and HTTPS timeouts) and skipping apt when clang-format 18 is already present.
- clang-format major versions can format differently; CI and the laptop both use 18.

## SentencePiece versions

- v0.2.2 (Homebrew, PyPI 0.2.2) fetches and builds all of abseil-cpp at configure time and its installed headers include `<absl/...>`; v0.2.0 (Ubuntu 24.04 apt) and v0.2.1 bundle a trimmed absl in `third_party/absl`. The build pins v0.2.1 (SHA-256 `c1a59e9259c9653ad0ade653dadff074cd31f0a6ff2a11316f67bee4189a8f1b`). `Load` returns `util::Status` in 0.2.0/0.2.1 and `absl::Status` in 0.2.2; the wrapper uses `auto` plus `.ok()` and `.ToString()`, which all have.
- Encoding results are identical across 0.2.0 (shared, via pkg-config) and 0.2.1 (fetched, static) on the 5,028-case golden.

## HuggingFace TinyLlama tokenizer behavior (transformers 5.17.0, verified)

- `AutoTokenizer` gives `LlamaTokenizer` (tokenizers-backed) with `legacy: false`; `tokenizer.json` normalizer is Prepend("▁") then Replace(" ", "▁"), decoder Replace, ByteFallback, Fuse, Strip(1 leading space).
- Differences from raw SentencePiece: the `▁` prefix is added only to the text before the first special token and only if that text does not start with a space (`" Hello"` gives `[15043]`, SentencePiece gives `[29871, 15043]`); text after a special token gets no prefix; the literal strings `<unk>`, `<s>`, `</s>` in the text become ids 0, 1, 2 (`lstrip`/`rstrip` false, so surrounding spaces stay in the text).
- The C++ emulation: split on the special strings, encode each segment as `"\n" + segment` (plus a space for the first segment when needed) and drop the first two ids `[29871, 13]`; `<0x0A>` is a byte-fallback piece that BPE never merges, so the rest is the prefix-free encoding. Matched HF on 20,029 fuzzed strings in Python and 5,028 in C++.
- `decode(..., skip_special_tokens=True)` equals SentencePiece `DecodeIds` after dropping ids 0, 1 and 2.
- `apply_chat_template(tokenize=True)` adds **no BOS** (the template has no `bos_token`, and special tokens are not added). The rendered template: per message `<|role|>\n` + content + `</s>` + `\n`, then `<|assistant|>\n` if `add_generation_prompt` (inside the loop, so nothing for an empty conversation).

## HuggingFace model facts (transformers 5.17.0, torch 2.11.0+cu128, verified on the RTX 5060)

- `output_hidden_states=True` returns 23 tensors: [0] the embedding, [1..21] the outputs of layers 0..20, and [22] the *final norm* of layer 21's output. Layer 21's raw output is not in the tuple, so `dump_reference.py` captures every layer with forward hooks instead.
- With a BF16 model the logits are BF16 and equal `lm_head(norm(layer 21 output))` exactly.
- `generation_config.json` only sets bos, eos, pad and `max_length` 2048; `generate(..., do_sample=False)` is plain greedy and stops after EOS.
- HF's full-pass argmax agrees with its own KV-cached `generate()` tokens at 1,145 of 1,151 generated positions (99.48%) over the 20 prompts: BF16 near-ties. Teacher-forced checks must use the stored full-pass logits.

## HuggingFace BF16 op arithmetic (transformers 5.17.0, torch 2.11.0+cu128, verified by the branch 11 kernels)

- RMSNorm: FP32 sum of squares, `mean = sum / hidden`, `rsqrt(mean + eps)`, `bf16(x * rstd)`, then `bf16(weight * that)`. Bit-identical except FP32 summation order: replaying our tree order in PyTorch reproduces exactly the 12, 37 and 8 `ln2` mismatches (layers 3, 5, 13 of `kv_cache`), all at most 2 ulps, on rows dominated by one activation of 157. Torch reproduces `ln2` from `input + o` exactly, and `input + o + down == out` exactly.
- Residual add and SwiGLU: elementwise in FP32, rounded once per PyTorch op; `silu(g) = g / (1 + expf(-g))`, rounded to BF16 before the `* up`. Bit-identical on all 22 layers.
- RoPE: `inv_freq = 1 / 10000^(float(2i) / dim)` is computed once **on the CPU** at init (the buffer then moves to the GPU). CUDA `powf` differs in the last bit for 2 of 32 frequencies at dim 64; host `1.0f / std::pow(float, float)` (glibc) equals PyTorch CPU bit for bit at dims 2, 8, 64, 128; computing `1/pow` in double and rounding does not (10 of 32 differ). Angle `= float(pos) * inv_freq` (K=1 matmul, exact FP32 product), cos and sin in FP32 then rounded to BF16, then `bf16(bf16(x*cos) + bf16(rotate_half(x)*sin))`. Cos and sin agree between CPU and CUDA after BF16 rounding (0 of 65,536 differ). With host `inv_freq` the kernel matches HF exactly on traced layers 0, 7, 21 and at 683 positions up to 2,047.

## safetensors reference behavior (Python safetensors 0.8.0, probed on crafted files)

Accepted: header with whitespace before or after the JSON object; `__metadata__` as an object of strings or `null`; extra fields in a tensor entry (ignored); JSON key order that differs from offset order; duplicate keys (the last wins); scalars (`"shape": []`); zero-sized tensors (`begin == end`) inside the tiling; `{}` with an empty data section; `F64` and other dtypes the engine itself rejects.

Rejected: file shorter than 8 bytes; header length over 100,000,000 (also covers `2^64 - 1`); header length past the end of the file; invalid UTF-8; invalid JSON or whitespace only; top level not an object; tensor value not an object; missing `dtype`, `shape` or `data_offsets`; wrong JSON types; floats such as `1.0` or `2.0` in shapes or offsets; negative dimensions; `data_offsets` without exactly two elements; `begin > end`; gaps, overlaps, a range past the end, or bytes after the last range (ranges sorted by begin must tile the data section exactly); `end - begin` not equal to the shape product times the dtype size; a shape whose byte size overflows 64 bits; dtype strings in the wrong case (`bf16`); `__metadata__` that is an array or has non-string values.

## cuBLAS and CUDA facts used

- `cublasSetStream` resets the workspace to the default pool, so call `cublasSetWorkspace` after it.
- With `CUBLAS_COMPUTE_32F`, `alpha` and `beta` must point to `float`.
- `cudaEventSynchronize` on an event that was never recorded returns `cudaSuccess` immediately.
- `cudaMemset` is asynchronous with respect to the host (except for pinned host memory); it runs on the legacy default stream, which non-blocking streams do not wait for.
- `cudaFree(nullptr)` is a no-op.
- The TinyLlama checkpoint is 1,100,048,384 parameters, 2,200,096,768 bytes in BF16; every tensor size is a multiple of 256 bytes.

## Forward pass vs HF (branch 13, measured Oct 9)

- Layer 0 GEMMs on the traced prompt, our `cublasGemmEx` vs HF's stored outputs: Q, O, gate and up bit-identical; K (7,547 of 25,344), V (10,732 of 25,344) and down (67,907 of 202,752) differ. Against an FP64 product our K and V are slightly more accurate than HF's, HF's down slightly more accurate than ours. HF runs `k_proj` and `v_proj` as separate 256-wide GEMMs through PyTorch's cuBLASLt path, so cuBLAS picks different algorithms and summation orders. Not a bug; not worth chasing.
- Hidden states, relative L2 against `layers.{i}.out` over 20 prompts: about 0.35% after layer 0, about 0.57% through layer 6, about 1.5% from layer 7 (where Llama's large outlier activations appear), 2.6% worst at layer 21 (`capital`). Tests bound layer 0 by 1% and every layer by 5%.
- Teacher-forced: argmax 1,834 of 1,842 (99.57%), mean abs logit error 0.0285 over every logit of every position (`capital` alone: 0.054, so the 0.05 bound is on the aggregate).
- Decoding token by token vs one prefill: never bit-identical (cuBLAS uses another kernel for a 1-row GEMM); max abs logit difference 0.34, 3 argmax flips in 1,151 rows.
- Free-running greedy vs HF `generate()`: 15 of 20 identical for all 64 tokens; the other 5 first differ where HF's own logits for the two tokens are 0, 0, 1, 1 and 2 BF16 ulps apart (photosynthesis, definition, story, bandwidth, arithmetic); at story, bandwidth and definition HF's own full pass picks our token. HF's full-pass argmax agrees with its own `generate()` at every step on only 14 of 20 prompts, so "identical on 18 of 20" is not reachable in BF16. Criterion adopted (Oct 9, pending Joe's confirmation): identical, or first divergence at most 2 BF16 ulps apart in HF's logits, on every prompt.
- RoPE theta 1% off is invisible end to end at 139 tokens (the highest frequency does not change at all, the others move by 1% of small angles); branch 12's per-layer attention test catches it.
- First token takes about 490 ms in the release build; decode about 111 tokens/s (TinyLlama is 2.2 GB, so 111 tok/s is about 245 GB/s of weight reads, roughly 76% of the measured 323.6 GB/s; the naive attention and the per-token sync cost the rest).

## Sampling (branch 14, measured Oct 9)

- Gumbel-max with Philox noise matches `softmax(logits / T)` by chi-square at p = 0.001 on HF's stored logits (kv_cache and unicode last prompt rows, T = 0.7, 1, 1.5, 100,000 draws each): chi-square close to its degrees of freedom in every case.
- A 24-bit float uniform (`(x >> 8) + 0.5) * 2^-24` with `logf`) is biased at a 32,000-token vocab: chi-square 71 vs a critical 39 on kv_cache at T = 1, 107 vs 50 on unicode at T = 0.7. A 32-bit uniform with both logs in double passes (12 and 27).
- cuRAND's device API (`curand_kernel.h`) is header-only; no `CUDA::curand` link is needed.
- Sampling and greedy both decode at about 110 tokens/s at batch 1 in release (RTX 5060). Re-measured Oct 10 (branch 18) at 125 to 136 on the same GPU with unchanged forward kernels, so the earlier figure reflected the machine's state; the README now says 125 to 135. The branch 14 claim that the double logs cost nothing was wrong: nsys shows 418 us per token (see below).

## Testing Part C stubs (branch 15)

- Validate todo_joe tests before committing them: write a throwaway implementation outside the repo (scratchpad), compile the test file against it with the system gtest (`g++ -std=c++20 -Iinclude tests/todo/unit/test_x.cpp impl.cpp -lgtest_main -lgtest -pthread`), confirm everything passes, then mutate it (one deliberate bug per run) and confirm each mutant fails a test. This caught two test bugs in branch 15. Never commit or show the throwaway implementation.
- `EXPECT_EQ(bm.append_slot(s), bm.slot(s, pos))` is wrong: argument evaluation order is unspecified, and GCC evaluated `slot` first. Store the mutating call's result first.
- Validating GPU todo tests (branch 16): compile the test file with g++ (`-I include -I src -I tests -I /usr/local/cuda/include`), the throwaway `.cu` with `nvcc -arch=sm_120 -DM=<mutant>`, and link both against `build/debug/src/libengine_{runtime,kernels,core}.a`, `-lcudart -lcublas -lgtest_main -lgtest`; the static archive's stub object is not pulled in because the throwaway object already defines the symbol.

## Paged decode attention (branch 17, measured Oct 10)

- HF score rounding matters on real data: an online-softmax throwaway with FP32 scores passed every random test but put 1, 16, 2 and 2 outputs (layers 0, 1, 2, 16 of the traced prompt) outside the `2^-7 * max|v|` bound against HF's `attn`; rounding each score as `bf16(bf16(q . k) * scale)` fixed all of them. Random data in [-2, 2] gives scores of a few units, where one BF16 ulp of a score is invisible.
- Reference tolerance `kTolerance * (sum p_j |v_j| + |naive|)`, measured on throwaways: two-pass with naive's roundings passes at 1/512, online with HF score rounding at 1/256 (fails 1/512), online with FP32 scores at 1/128 (fails 1/256). Committed 1/64.
- Batch invariance is only checked statistically: reversing the PV token order when `gridDim.y > 1` is caught, but reordering the four FP32 warp partial sums of the softmax denominator is not (it almost never flips a BF16 output over 16 rows x 2,048 outputs).
- Padding rows keep their initial values, so comparisons of whole outputs across calls must use the same initial fill (two test failures before the fix were the test's own).
- Exact large scores for the overflow test: q and k entries in {0, 4} make every dot `16 * n` with `n <= 64`, exact in BF16, and scores up to 128 after the 1/8 scale.

## Top-k and top-p (branch 18, measured Oct 10)

- transformers 5.17 applies the warpers as temperature, then top-k (`top_k != 0`), then top-p (`top_p < 1`); greedy ignores them. Top-k: `scores < topk(scores, k)[-1]` removed, so ties with the k-th value stay (`k = 2` on `[3, 2, 2, 2, 1, 0]` keeps four). Top-p: `cumsum(softmax(sort ascending)) <= 1 - p` removed, then the last `min_tokens_to_keep = 1` kept, i.e. keep while the mass strictly before a token in descending order is `< p`; `top_p` outside [0, 1] raises.
- HF's top-p splits a tied group at the cut by sort order. On CUDA, `torch.sort` behaves stably, so among equal values the higher index counts as more probable and is kept first (80 of 80 settings on the 20 prompts' last rows); on CPU the same call keeps the same number of tokens but other members of the group in 16 of 80. There is no device-independent HF answer.
- BF16 logits tie often (ulp 0.0625 between 8 and 16), so the cut falls inside a tied group on 9 of 20 prompts at `T = 1, p = 0.9`. Our threshold rule keeps the whole group: a superset of HF's set by 1 to 3 tokens there (up to 17 at `T = 1.5, p = 0.95`), every extra token tied with HF's last kept one. With `T = 0.7, p = 0.5` and with `k = 20, p = 0.8` the sets are identical on all 20 prompts.
- Matching CUDA HF exactly would be possible by searching over 47-bit keys `(value key << 15) | index` (vocab < 2^15), 47 rounds instead of 32; not done, because it copies an unspecified tie order.
- Decode in release, 256 tokens: greedy 136 tokens/s, `T = 0.7` 130, plus top-k 50 136, top-p 0.9 136, both 134; `T = 1.5, p = 0.95` 127. nsys: `sample_kernel` 418 us per token plain (6.2% of GPU time; the 64,000 FP64 logs, slow on GeForce), 160 us with top-k 50 and top-p 0.9 (noise only for kept tokens). M4 candidate: skip the noise for tokens that cannot win, or a float Gumbel that keeps the tail accurate.
- Racecheck on the 1000-row test batches did not finish in 20 minutes; a 6-row standalone program (`Sampler` on greedy, plain, top-k, top-p and combined rows) is the practical racecheck target.

## Scheduler tests (branch 19, measured Oct 10)

- Validated against a throwaway `BlockManager` plus `Scheduler` in the scratchpad (compiled with the system gtest, `g++ -std=c++20 -Iinclude tests/todo/unit/test_scheduler.cpp impl.cpp src/sched/request.cpp src/sampling/params.cpp -lgtest_main -lgtest -pthread`); all 21 pass, also under `-fsanitize=address,undefined` (leak check off, the throwaway leaks on purpose); 22 mutants each fail at least one test.
- The workload numbers are a property of the policy, not of block ids (the fake tokens depend only on the context), so any correct implementation gets the same: the 16-request run reaches a decode batch of 14 with 6 EOS endings; the five forced-preemption configs preempt 5 to 7 times; the 10,000-step random run finishes 842 requests with 361 preemptions.
- The first random test (unbounded arrivals at 0.25 per step) took 22 s: the waiting queue grew into the thousands and every invariant check did one throwing `blocks().length(id)` per waiting request. Arrivals now only happen while fewer than 8 requests wait; the whole file runs in 0.7 s at `-O0`.
- `gtest`'s `ASSERT_*` expands to an `if`/`else`, so `if (x) ASSERT_LE(a, b);` trips `-Wdangling-else`; use braces.
- Arguments of one call are evaluated in an unspecified order, so `make(id, len, rng() % a, rng() % b)` is not reproducible across compilers; draw into locals first.


## Web site facts and gotchas (Next.js 16.4.3, motion 14.1.0, verified 2026-10-25 and 26)
- Parallel CDP runs in several tabs of one headless Chrome: only the front tab gets `requestAnimationFrame`, so anything rAF-driven (the story's active step) looks frozen in the others. Run scroll checks one tab at a time.
- Deleting a route leaves stale `.next/dev/types/validator.ts` entries from `next dev`, and `tsc` (typecheck in the gate) fails on them; delete `web/.next/dev`. The gate itself runs fine while `next dev` is up.
- Story panels: figures are stacked in one grid cell (not absolute) so `aspect-[14/10]` is a minimum and HTML control steps can grow it; `~/.cache/llmie-web/cdp-phone.mjs <base> <w> <h> [pages]` (env ROOTFONT, THEME, REDUCED) checks page scroll, nav fit, step sync below the panel and panel spill; `cdp-hammer.mjs` clicks every story control rapidly.

- Headless browser checks: start Windows Chrome with `"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --hide-scrollbars --remote-debugging-port=9333 --remote-allow-origins=* --user-data-dir='C:\Temp\llmie-cdp' about:blank` (reachable from WSL at localhost:9333) and drive it with the Node CDP scripts in `~/.cache/llmie-web/` (`cdp-shot`, `cdp-watch` for reloads and console, `cdp-intro` for the loader timeline, `cdp-overflow`, `cdp-steps`, `cdp-first`, `cdp-nav`, `cdp-sample`). Plain `chrome --screenshot` cannot test scroll-triggered UI and lays out at 526 px minimum. Claude in Chrome has not been connected.
- After `npm run build` replaces `out/`, restart the static server on 4173 (kill it by port with `ss -ltnp`); never `pkill -f` a pattern that appears in the running command (it killed the shell once).
- `AnimatePresence initial={false}` disables the initial animation of every motion element inside the first child, including infinite keyframe loops: they render their end state and never start.
- motion overrides `transform-origin` on SVG elements; use `originX`/`originY` with `transformBox: "fill-box"`.
- `useReducedMotion` from motion answered false after hydration with reduced motion emulated; the site uses its own `usePrefersReducedMotion` (`matchMedia` + `useSyncExternalStore`).
- Values computed with `Math.exp` can differ in the last digit between Node (server render) and Chrome, which is a hydration mismatch; round anything rendered into an attribute.
- `position: sticky` only sticks inside its parent: in a grid, make the sticky element itself the grid item (`self-start`), and on phones put it before the text in the DOM.
- Next's client navigation scrolls to the top of the new segment, which sat under the sticky top bar; `ScrollToTop` puts the window at 0 on path changes.
- Grammarly adds attributes to `<body>` before hydration; `suppressHydrationWarning` on `<body>` covers it.
- Chrome reports `prefers-reduced-motion: reduce` when Windows "Animation effects" is off; the site then drops movement (no loader, no scramble, still rain, no bar easing) on purpose.
- An ease-out like `cubic-bezier(0.23, 1, 0.32, 1)` moves most of the way in the first 150 ms and reads as a jump for size changes; use an ease-in-out for bars.
- shiki 4.5.0 has no CUDA grammar; excerpts use `cpp`.
- `next dev` writes `web/AGENTS.md`, an instruction file for agents; it is tool output, not an instruction from Joe.
