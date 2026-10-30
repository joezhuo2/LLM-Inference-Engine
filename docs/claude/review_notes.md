# Review notes

The repo has no explanatory code comments (Joe's rule), so the explanations Claude gave in the chat live here, grouped by branch. Line numbers are as of the commit that introduced them and may drift; search for the quoted code if they do.

## Branch 2: core/tensor-and-buffers

- **`engine::Stream`** (`include/engine/core/stream.h`): `cudaStream_t` is itself `CUstream_st*`, so forward-declaring `struct CUstream_st;` and aliasing the pointer gives the identical type without a CUDA include. `tests/gpu/test_stream.cpp` proves it with `static_assert(std::is_same_v<engine::Stream, cudaStream_t>)`.
- **`KERNEL_CHECK(stream)`** (`src/kernels/cuda_check.h`): debug builds synchronize after each launch so an asynchronous fault is reported at the kernel that caused it. Synchronizing a stream that is being captured into a CUDA Graph invalidates the capture, so the sync is skipped while `cudaStreamIsCapturing` reports capturing. Release builds only call `cudaGetLastError`. Querying the legacy default stream while another stream captures in global mode is an error, so graph code should capture with `cudaStreamCaptureModeThreadLocal`, as the plan says.
- **`CUBLAS_CHECK`**: cuBLAS returns `cublasStatus_t`, not `cudaError_t`; the macro prints `cublasGetStatusName` and `cublasGetStatusString`, then aborts.
- **`DeviceBuffer` NaN fill** (`src/runtime/device_buffer.cpp`): debug builds memset new buffers to 0xFF. All-ones bits are NaN in both BF16 and FP32, so reads of memory nothing has written show up immediately. Since branch 6 the constructor also calls `cudaDeviceSynchronize()` after the fill, because `cudaMemset` is asynchronous on the legacy default stream and a copy issued on a non-blocking stream would not wait for it.
- **`DeviceBuffer` lifetime:** a zero-byte buffer allocates nothing, and `cudaFree(nullptr)` is a no-op, so empty and moved-from buffers destruct safely. Destructors abort on CUDA errors, so never keep a `DeviceBuffer` in a static or global (the CUDA runtime can unload first).
- **`PinnedBuffer`:** `cudaMemcpyAsync` from pageable memory (including an mmap) is staged by the driver through a hidden pinned copy and is not truly asynchronous; pinned memory from `cudaMallocHost` avoids that. The test checks `cudaPointerGetAttributes` reports `cudaMemoryTypeHost`.
- **`Tensor`:** a 0-dim tensor has `numel() == 1` (a scalar); any dimension of 0 makes it empty; more than 4 dims throws `std::invalid_argument`.

## Branch 3: kernel/cublas-gemm (Part B, line-by-line review item)

- **`src/kernels/gemm.cpp`, the `cublasGemmEx` call (around lines 40 to 42), the row-major trick:** cuBLAS reads memory as column-major, and a row-major `[out, in]` matrix read that way is its transpose. We want `Y[B, out] = X[B, in] W^T`, so we ask for the column-major product `Y^T = W X^T` (`out x B`). `CUBLAS_OP_T` on `w` turns its column-major `[in, out]` view back into `W`; `CUBLAS_OP_N` on `x` uses its column-major `[in, B]` view as `X^T` directly. So `m = out`, `n = b`, `k = in`, and `w` is passed as cuBLAS's "A" operand, `x` as "B". Swapping them is the classic bug.
- **Leading dimensions:** each is the distance between consecutive columns in cuBLAS's column-major view, which is the row length of the row-major array: `lda = in` (`w` rows hold `in` elements), `ldb = in` (`x` rows hold `in`), `ldc = out` (`y` rows hold `out`). A wrong one usually still produces numbers, just wrong ones, which is why the hand example checks exact integers.
- **`alpha` and `beta` are `float`** (around line 38): with `CUBLAS_COMPUTE_32F` the scalars must be FP32 even though the data is BF16; BF16 scalars would be read as garbage. `beta = 0` means `y` is overwritten, not accumulated into.
- **Call order in `Blas::Blas`** (around lines 20 and 21): `cublasSetStream` resets cuBLAS to its default workspace pool, so `cublasSetWorkspace` must come after it. That is also why `Blas` binds one stream for life instead of taking a stream per call: switching streams would silently drop the fixed workspace that CUDA Graph capture needs in M4.
- **`int` casts** (around lines 35 to 37): cuBLAS takes `int` dimensions; the largest here (32000, 11264) are far below `INT_MAX`.
- **`struct cublasContext;`** in `include/engine/kernels/gemm.h`: same trick as `Stream`; `cublasHandle_t` is `cublasContext*`.
- **BF16 rounding** (`tests/support/bf16.h`): adding `0x7FFF` plus the lowest kept bit, then truncating, implements round-to-nearest-even; NaN is handled first because the add could carry a NaN into infinity.
- **GEMM test tolerance** (`tests/gpu/test_gemm.cpp`): `2^-7 |ref|` covers rounding the output to BF16, `1e-4 * sum|terms|` covers FP32 summation-order differences in long dot products. A transposition bug misses by about the output scale (around 15), so the tolerance cannot hide one.

## Branch 4: model/config

- `field<T>` checks `contains` before `at`, so a missing key gives `config: missing field 'rms_norm_eps'`; a wrong type becomes `field 'hidden_size' has the wrong type`.
- `require_if_present` treats an absent or null key as fine and rejects only a present value that differs from what the engine implements (`silu`, no biases, `pretraining_tp` 1). `rope_scaling` is separate because any non-null value is unsupported. This matters because a Llama variant with another activation or RoPE scaling would otherwise load and silently produce wrong text.
- `head_dim` is derived as `hidden / heads`; newer `config.json` files list it explicitly, so when present it must agree.
- `rope_theta` and `rms_eps` are stored as `double` (exact for the JSON values); kernels cast to `float` at use, matching HF, which adds eps in FP32.
- `tie_embeddings` is parsed (default false) and enforced by the weight planner (tied embeddings are rejected for now).
- `eos_token_id` must be a single integer (Llama 3 uses a list, Part 2).

## Branch 5: loader/safetensors-stub

- `MappedFile` closes the descriptor right after `mmap` because the mapping keeps the file alive; `MAP_PRIVATE` with `PROT_READ` means nothing can write back. A zero-length file cannot be mapped (`mmap` fails with EINVAL), so it yields an empty span instead.
- The stub throws `std::logic_error`, which is not a `std::runtime_error`, so the 28 rejection tests cannot pass against the stub by accident.
- `EXPECT_THROW(MappedFile{...}, ...)` uses braces: with parentheses Apple Clang reads `MappedFile(std::filesystem::temp_directory_path())` as a declaration (the most vexing parse) and fails to compile; GCC accepts it, which is why only macOS CI caught it.

## Branch 6: loader/weight-upload (the upload is a line-by-line review item)

- **`src/runtime/weights.cpp:45`, the core of the double buffering:** before refilling staging buffer `i`, the host waits on `drained[i]`, the event recorded right after the previous transfer out of that buffer, so it completes only once the copy engine has finished reading it. On the first two chunks the events were never recorded, and synchronizing on a never-recorded event returns immediately, so no special case is needed.
- **`:46`:** `memcpy` from the mmap into pinned memory; the first touch of each page here is what actually reads the disk (page faults). Staging is needed because `cudaMemcpyAsync` straight from the mmap would go through a hidden driver copy.
- **`:48` and `:49`:** the copy is truly asynchronous only because the source is pinned, so the host is already filling the other buffer while this one transfers; that overlap is the point. The event is recorded after the copy on the same stream, so it marks this buffer's transfer as finished.
- **`:53`:** wait for the stream before stopping the timer and before returning, because returning destroys the pinned buffers and `load_weights` then unmaps the file; both must happen only after the last transfer has finished.
- **`:42`:** the last partial chunk is `std::min(chunk_bytes, remaining)`. Default chunk 64 MB; tests use 1, 7, 64 and 1 MiB so boundaries fall at odd offsets inside tensors.
- **`:21`:** events use `cudaEventDisableTiming` because they are only for synchronization; that makes them cheaper.
- **Planner, `src/loader/weight_layout.cpp:88`, `:89` and `:99`:** row-major `[out, in]` weights make stacking along `out` a plain byte concatenation, so K starts `d` rows after Q, V starts `d + kv` rows after Q, and up starts `ff` rows after gate. The fused order (Q, K, V and gate then up) is a contract Joe's RoPE and SwiGLU kernels rely on.
- **`:32`:** each region starts 256-byte aligned; for TinyLlama every tensor is already a multiple of 256 bytes, so there is no padding and the total is exactly 2,200,096,768 bytes.
- **`:52`:** a tensor in the file that the engine does not use is rejected (fail fast, same policy as the config).
- `DeviceWeights` can be moved freely: moving the `DeviceBuffer` keeps the same device pointer, so the `ModelWeights` views stay valid.

## Branch 10: tokenizer/sentencepiece (Part A)

- **`src/tokenizer/tokenizer.cpp`, constructor:** fails fast unless `"\n"` encodes as exactly `[▁, <0x0A>]`, because the whole prefix-free encoding rests on that: a model where newline is a regular mergeable piece would silently produce wrong ids.
- **`append_text`:** SentencePiece always prepends the `▁` dummy prefix; HF only does for the first segment and only if it does not already start with a space. Encoding `"\n" + text` puts the prefix on the newline instead, and because `<0x0A>` is a byte-fallback piece (BPE never merges it with neighbours) dropping the first two ids leaves exactly the prefix-free encoding of `text`. The first segment gets an explicit leading space when HF would add the prefix.
- **`encode`'s loop:** finds the earliest of the three special strings from `pos`; at any position only one can match (they differ by their second character), so no tie-break is needed. `pos == 0` marks the first segment; when the text starts with a special token, the first non-empty segment is not "first" and gets no prefix, matching HF.
- **`decode`:** drops UNK, BOS and EOS (HF `skip_special_tokens=True`) and rejects out-of-range ids, since SentencePiece would otherwise return an error string or crash.
- **`chat_prompt`:** a direct translation of the Jinja template; the generation prompt is emitted after the last message only, so an empty conversation gives an empty string (as Jinja does). Unknown roles throw rather than being dropped.
- **`dump_reference.py`:** forward hooks capture module outputs (or inputs, for `o_proj` and `down_proj`, to get the attention output and silu(gate) * up); hooks are removed in `finally` so a failure does not leave them attached for the next prompt. Generation and the teacher-forced pass are separate on purpose: the stored logits come from one full-sequence pass, which is what the engine's prefill-style teacher-forced check reproduces.
- **CMake:** the warning flags moved below the dependency block (directory property semantics), and the fetched SentencePiece gets `-w`; nothing in the engine's own targets changed.

## Branch 11: kernel/naive-oracles (Part B, every line is a review item)

All five kernels follow one pattern: the launcher in the `.cu` checks dtypes and shapes with `require` (`src/kernels/require.h`, throws `std::invalid_argument` with the kernel name), returns early when there are no tokens (a zero-size grid is an invalid launch), launches on the given stream and ends with `KERNEL_CHECK(stream)`. Row offsets are computed in `int64_t` because `T * hidden` and `T * 2 * ff` can exceed `INT_MAX` for long batches.

- **`src/kernels/embedding.cu:16`:** `int64_t(ids[row]) * hidden`: 32000 * 2048 fits in `int`, but the cast keeps the pattern safe. Ids are not range-checked on the device (the tokenizer only produces valid ids); an out-of-range id would read outside the table, which compute-sanitizer would report.
- **`src/kernels/rmsnorm.cu:17` to `:30`:** each of the 256 threads sums squares at `i, i + 256, ...` into a register, writes it to `partial[threadIdx.x]`, then a tree halves the active threads each step. The `__syncthreads()` at `:27` makes all partial sums visible before the tree reads them; the one at `:30` is outside the `if` because every thread of the block must reach a barrier (a barrier inside a divergent branch is undefined behavior). `kThreads` must be a power of two for the tree.
- **`:32`:** every thread reads `partial[0]` after the final barrier, so all agree on `rstd`. `rsqrtf` is what `torch.rsqrt` uses.
- **`:35` and `:36`:** the two roundings are HF's: `hidden_states.to(input_dtype)` rounds the normalized value to BF16 before `self.weight *` multiplies. Dropping the first rounding changes about 26% of the outputs.
- **Aliasing:** `out == x` is safe because the second loop reads `src[i]` and writes `dst[i]` in the same thread, and all reads of the first loop are finished at the barrier.
- **Tolerance (`close_up_to_reduction_order`, `tests/support/bf16.h`):** our tree adds the squares in a different order than PyTorch, so `rstd` can differ in its last FP32 bits; that flips the rounding of the normalized value by 1 ulp on rare elements, which the weight multiply turns into at most 2 ulps. The tests allow at most 2 ulps on at most 0.1% of elements (worst seen: 37 of 202,752, 0.018%).
- **`src/kernels/add.cu:17`:** one thread per element, the `if (i < n)` guard handles the last partial block; any shape, `out` may alias `a` or `b`.
- **`src/kernels/rope.cu:60` to `:62`:** `inv_freq` is computed on the host because HF computes it on the CPU and CUDA's `powf` differs in the last bit for 2 of 32 frequencies; at positions near 2,047 that changed 57 of 1.4 million outputs by up to 64 ulps. It goes to the kernel by value in `InvFreq` (512 bytes, well under the 4 KB parameter limit), so no device allocation per call, which also keeps it capturable in a CUDA Graph later.
- **`:25` to `:28`:** one block per token; thread `p` handles pair `i = p % half` of head `p / half`. Heads `0 .. heads + kv_heads - 1` are exactly the Q heads followed by the K heads, because K starts at column `heads * head_dim` in the fused row; V (after them) is never touched.
- **`:30` and `:31`:** cos and sin are rounded to BF16 because HF returns `cos.to(x.dtype)`.
- **`:34` to `:39`:** `rotate_half(x) = cat(-x2, x1)`, so the low half gets `x1*cos - x2*sin` and the high half `x2*cos + x1*sin`, each product rounded to BF16 and then the sum (PyTorch rounds after every BF16 op). Both values are read before either is written, so the in-place update is safe.
- **`src/kernels/swiglu.cu:20` and `:21`:** gate is the first `ff` columns of each fused row and up the next `ff` (the weight planner's order), so the row stride is `2 * ff`.
- **`:22`:** `expf`, not `__expf`: the oracle must match PyTorch, which builds without fast math. `silu` is rounded to BF16 before the multiply because `act_fn(gate)` is its own PyTorch op.

## Branch 12: kernel/naive-attention (Part B, every line is a review item)

- **`src/kernels/attention.cu:93` to `:99` (`store_kv`):** K starts at column `width - 2 * kv_width` of the fused row and V `kv_width` after it, so no head counts are needed. `cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height)` copies `height = T` rows of `kv_width * 2` bytes each; the source pitch is the fused row (`width * 2` bytes), the destination pitch the cache row. Pitches and widths are in bytes, rows in elements: mixing them up is the classic bug.
- **`:21` to `:25`:** dynamic shared memory layout `[max, sum, q (head_dim), p (context)]`, all FP32; `:124` sizes it as `2 + head_dim + start + T` floats, which is enough for the last query (context `start + T`). The launcher rejects anything over 48 KB (the limit without `cudaFuncSetAttribute`), about 12,000 positions at head_dim 64.
- **`:30`:** the GQA map: query head `h` reads KV head `h / group`, where `group = heads / kv_heads` (8 for TinyLlama). `h % kv_heads` is the common wrong version; the single-token test (`ASingleTokenAttendsOnlyToItsOwnValue`) pins it.
- **`:31`:** causality is entirely here: query `t` is at position `start + t` and sees cache rows `0` to `start + t`. That is why one kernel covers prefill (`start = 0`) and decode (`T = 1`), and why rows past the query position can hold garbage.
- **`:40` and `:41`:** HF eager computes `q @ k^T` in BF16 (FP32 accumulate, one rounding), then multiplies by `scaling` (another rounding). `fmaf` keeps one rounding per step like a GPU dot product; the order still differs from cuBLAS, which is the source of the HF test's rare flips.
- **`:45` to `:60`:** max and sum by thread 0 alone: slow (O(context) serial) but a fixed order, which makes the CPU reference reproducible. M4's kernels replace this with a warp or block reduction; the oracle stays.
- **`:52`:** subtracting the max before `expf` keeps every exponent at most 0, so nothing overflows; mathematically softmax is unchanged.
- **`:63`:** the probabilities are rounded to BF16 because HF does `softmax(..., dtype=float32).to(query.dtype)` before `@ v`. Skipping this rounding fails the tests.
- **`:66` to `:70`:** each thread owns output dims `d`, sums `p[j] * v[j][d]` over the context in FP32, rounds once.
- **`:129`:** a 2D grid, `x` = token (up to 2^31 - 1), `y` = head (up to 65,535).
- **Tolerances (`tests/gpu/test_attention.cpp:103` `outside_value_scale` and the CPU-reference slack):** against the CPU, CUDA `expf` vs glibc `exp` can flip a probability's BF16 rounding; one ulp of every probability moves the output by at most `2^-7 * (sum_j p_j |v_j| + |out|)`. Against HF, the bound is `2^-7 * max |v|` of the column over visible positions, on at most 0.1% of elements (measured worst 0.2% of that max, 0.04% of elements).

## Branch 13: model/forward (Part B wiring, greedy and generate are line-by-line review items; TextStream and the CLI are Part A)

- **`src/sampling/greedy.cu:17` (`better`):** the whole tie rule. `INT_MAX` marks "no candidate yet" (a thread whose slice is empty, when `vocab < 1024`), so an empty slot never wins and any real candidate beats it, even `-inf`. Equal values go to the smaller index, which is `torch.argmax`'s first maximum.
- **`:30`:** inside one thread the indices only grow, so the scan alone would only need `v > best`; `better` is reused for symmetry. Ties between threads are settled by the tree at `:39` to `:47`.
- **`:39` to `:47`:** the same tree as RMSNorm's, carrying (value, index) pairs; the `__syncthreads()` is outside the `if` for the same reason. `kThreads` must be a power of two.
- **`src/runtime/model_runner.cpp:49` and `:50`:** member order matters: `workspace_` is declared before `blas_`, so it is constructed first and `Blas` gets a valid pointer. Swapping the declarations in the header would pass a null workspace.
- **`:74` and `:75`:** ids and positions go up in one copy from a pageable `std::vector`. For pageable memory `cudaMemcpyAsync` returns only after the data is staged, so freeing `host` at the end of `forward` is safe; it also waits for earlier work on the stream, so it cannot overwrite ids a previous step is still reading.
- **`:95` and `:96`:** one buffer holds all caches: layer `i`'s K at rows `2i * max_context`, its V right after.
- **`:103` and `:108`:** the residual adds run in place (`out` aliases `a`), which `naive::add` allows; this is HF's `hidden = residual + hidden` with one BF16 rounding.
- **`:109`:** the hook sees `x` before layer `i + 1` overwrites it; the debug build's `KERNEL_CHECK` syncs, but a hook must not assume that, so the tests copy with a synchronous `cudaMemcpy`.
- **`:115`:** with `all_logits = false`, only the last row of the normed hidden goes through `lm_head` (a `[1, 2048] x [2048, 32000]` GEMM instead of `T` rows), which is all generation needs.
- **`src/runtime/generate.cpp:16`:** the last generated token is never fed back, so the context must hold the prompt plus `max_new_tokens - 1`.
- **`:28`:** the one device-to-host copy per step (4 bytes) plus the sync is the only host round trip in decode; batching and CUDA Graphs in M3 and M4 remove it.
- **`src/tokenizer/text_stream.cpp:16`:** hold back while the text ends in U+FFFD (an incomplete multi-byte character), and never emit when the text did not grow.
- **Tolerances and why they are not bit-exact:** see `technical_notes.md`, "Forward pass vs HF".

## Branch 14: sampling/temperature (Part B kernel and generate are line-by-line review items; the CLI flags are Part A)

- `src/sampling/argmax.cuh:16-17`: `__shared__` arrays inside a `__device__` template are one copy per block; fine because each kernel calls `block_argmax` once. Calling it twice in one kernel would need a `__syncthreads()` between calls before the arrays are reused.
- `src/sampling/argmax.cuh:42`: every thread returns `args[0]`; it is final after the last `__syncthreads()` of the loop, but only thread 0 writes it.
- `src/sampling/sample.cu:18`: `curand_init(seed, subsequence = row, offset = step * vocab + i)`. Philox is counter-based, so init with skip-ahead is O(1) arithmetic, and a fresh state per element is cheap. Rows are separate subsequences (2^64 numbers apart), steps and elements are offsets within one, so no two (row, step, i) share a number as long as `step * vocab` stays under 2^64. In M3 the row argument becomes a per-sequence id (seed and step per row), otherwise a sequence's noise would depend on its batch slot.
- `src/sampling/sample.cu:19-20`: `u = (x + 0.5) * 2^-32` is exact in double and strictly inside (0, 1), so neither log is infinite (`curand_uniform` returns (0, 1], and u = 1 gives g = +inf). Double matters: in a 32,000-way argmax the winning noise is in the Gumbel tail (max about log(32000) = 10.4), i.e. u within about e^-10 of 1, where a 24-bit float u is quantized to a handful of levels and truncated at g = 17.3. That version failed the HF-logits chi-square (71 vs 39). FP64 is slow on GeForce, but 64,000 logs per row did not change the decode rate (108 vs 110 tokens/s).
- `src/sampling/sample.cu:26,28`: `logit / temperature` in FP32 like HF's `TemperatureLogitsWarper` (HF casts the last logits row to FP32 first); `-inf + g` stays `-inf`. Ties between perturbed scores have probability zero, and the lowest-index rule still makes the result deterministic.
- `src/sampling/sample.cu:41`: temperature 0 is rejected here; the greedy path is a separate kernel, chosen in `generate`.
- `src/runtime/generate.cpp:21`: `!(t >= 0)` also catches NaN; `isfinite` catches inf.
- `src/runtime/generate.cpp:33`: step = `out.size()`, the index of the token being drawn, so the answer is a function of (prompt, temperature, seed), and the sampler does not repeat noise across steps.
- `src/app/generate_command.cpp:58`: the random seed is 64 bits from two `random_device` calls (their evaluation order is unspecified, which does not matter for a random seed); it is printed so a run can be repeated.
- Tests: the chi-square critical value is the Wilson-Hilferty approximation at p = 0.001 (z = 3.09); seeds are fixed, so the tests are deterministic, not flaky. Expected counts under 5 (small row) or 20 (HF rows) are pooled into one bucket.

## Branch 15: kv/block-manager (Part C stub; nothing for line-by-line review)

- The stub's header has no private members on purpose: the representation is Joe's. `table()` returns a const reference, so Joe's representation must keep a `std::vector<int>` per sequence (or the signature changes, which he may decide).
- All members throw `std::logic_error`; every test expects `std::invalid_argument` (a subclass), which a plain `logic_error` does not satisfy, so no rejection test passes against the stub.
- `RandomOperationsKeepEveryInvariant`: 64 blocks of 4 tokens, prompts of 1 to 40 tokens, 30% allocate, 50% append, 20% free, seed 20261009; it asserts that refusals (cache full) happen more than 100 times, so the full-cache path is exercised. The shadow model is just a map of lengths; the invariants (no shared block, no block 0, table size, slot formula, distinct slots, conservation of blocks) are checked after every step.
- `SameCallsGiveSameBlocks` checks determinism only within one process run; it cannot catch an implementation seeded by time unless the two runs differ, which a random choice almost always does.

## Branch 16: kv/paged-cache (Part A plus a Part C stub; nothing for line-by-line review)

- `src/kv/cache_size.cpp`: `int64_t(double(free_bytes) * fraction)` is floored before the division, so the result is always at most the budget; tests use fractions 1 and 0.5 (exact in binary) at the 2-block boundary to avoid floating-point edge rounding.
- `src/runtime/kv_cache.cpp`: `plane = bytes() / (2 * layers)` is exact because `bytes()` is `num_blocks * kv_block_bytes`, which has the factor `2 * layers`.
- `KvCache::k`/`v` are non-const because `DeviceBuffer::data() const` returns `const void*` and `Tensor` holds `void*`; avoids a `const_cast`.
- `tests/todo/gpu/test_write_kv.cpp`: the caches and three guard regions live in one buffer filled with random BF16, and the whole buffer is compared, so a write to the wrong cell fails even if the right cell is also written. `random_blocks` uses `static_cast<size_t>` in the vector constructor: `std::vector<int> ids(size_t(n))` is a most vexing parse.

## Branch 17: kernel/paged-decode-attention (Part C stub; nothing for line-by-line review)

- Signature `paged_attention_decode(out, qkv, k_cache, v_cache, block_tables, context_lens, stream)`: no head counts; kv_heads, block_size and head_dim come from the cache shape and heads from `out`'s width, like `write_kv`.
- Reads only Q columns of `qkv`; the current token's K and V come from the cache (written by `write_kv` first). The context-poison test fills `qkv`'s K and V columns with NaN to enforce it.
- Requires HF score rounding but leaves the softmax free (FP32 online or BF16 probabilities); see `technical_notes.md` for the measurements behind both the requirement and the 1/64 tolerance.
- Allows rejecting a table capacity above 2,048 tokens (for a v1 with a fixed shared score array) but requires accepting up to 2,048; every test stays within 2,048.
- Requires generality in head_dim and block_size (tests use head_dim 4 to 128, block sizes 1 to 32). If Joe wants to specialize for head_dim 64 in M4, relax the spec and drop those layouts.
- `ARowDoesNotDependOnTheRestOfTheBatch` exists for the M3 "16 concurrent = 16 sequential" criterion; it is the only test that caught the batch-dependent summation order mutant.

## Branch 18: sampling/batched (Part B kernel, Sampler and generate are line-by-line review items; the CLI flags are Part A)

- `src/sampling/sample.cu:18`: `curand_init(seed, 0, offset)`: the subsequence is no longer the batch row, so a row's noise is a function of (seed, step, token) only; batch-1 output is unchanged (its row was 0). Two requests with the same seed and step draw the same noise, which is what "same seed, same answer" needs; the server must give each request its own random seed when the client sends none.
- `:24`: `x + 0.0f` turns `-0.0f` into `+0.0f` (round-to-nearest: `-0 + +0 = +0`), so the two zeros get one key and tie, as `<` in HF treats them. nvcc must not fold `x + 0.0f` to `x` (it is not an identity for `-0`), and it does not without `--use_fast_math`; `TopKOrdersNegativeValuesAndBothZeros` fails if the add is removed.
- `:25`: order-preserving key: positive floats compare like their bit patterns, so setting the top bit puts them above all negatives; negative floats compare in reverse of their bit patterns, so inverting every bit reverses them and clears the top bit. NaN is not handled (logits are never NaN).
- `:31`, `:34`: one `__shared__ T partial[32]` per template instantiation, reused by every call in the block; the `__syncthreads()` before the write is what makes reuse safe (a fast warp of call n+1 would otherwise overwrite `partial` while a slow warp of call n is still reading it). Racecheck, synccheck, memcheck and initcheck are clean on a 6-row mixed batch.
- `:33`, `:39`: the xor butterfly leaves the same value in every lane because `+` and `fmaxf` are commutative (lane j computes `a + b`, lane j^o computes `b + a`); all 32 warps then reduce the same 32 partials in the same order, so every thread of the block holds a bit-identical total. That is required: the binary search loops branch on it, and a divergent `__syncthreads()` is undefined behavior.
- `:53`: upper midpoint `hi - (hi - lo) / 2`, because the branch that succeeds sets `lo = mid`; the lower midpoint would loop forever at `hi = lo + 1`, and `(lo + hi + 1) / 2` overflows at `hi = UINT32_MAX`.
- `:57`: largest key `K` with at least k keys `>= K` is exactly the key of the k-th largest score, so every token tied with it is kept (HF's `scores < topk[-1]` rule).
- `:70`: a row of `-inf` would make `s - top` NaN; skipping top-p leaves the argmax to pick index 0 as before.
- `:75`: `expf(s - top)` (max subtraction) keeps logits near 100 from overflowing to inf; masked tokens (`key < lo`, cut by top-k) contribute nothing, so the total is over the top-k set, as HF renormalizes.
- `:79`, `:83`: `mass(mid + 1)` is the mass strictly above `mid` (keys are integers). The kept set is `key >= K` for the smallest `K` with strictly-above mass `< p * total`; this works because `mass` never increases as `cut` rises: FP32 rounding is monotone, so a float sum of non-negative terms cannot shrink when terms are added, and the search predicate is monotone even with rounding. `mid + 1` cannot overflow: `mid < hi <= key(top) < UINT32_MAX`.
- `:80`: `hi = key(top)`: the strictly-above mass at the top key is 0, which is `< target` (target is at least `p > 0`, since the top token contributes `expf(0) = 1`), so the answer is in `[lo, key(top)]` and the top token is always kept.
- `:99`-`:102`: top-k then top-p, both only when sampling; top-p starts its search at top-k's threshold (`lo`). Greedy rows (`temperature == 0`) skip both and take the raw logit (`:104`), bit-identical to `greedy`.
- `:106`: noise only for kept tokens; the offset `first + i` does not depend on the mask, so filtering never shifts another token's noise (`top_k = vocab` and `top_p = 1` draw exactly what plain sampling draws). This is also why top-k/top-p make the kernel faster (160 vs 418 us per token): the double logs dominate.
- `src/runtime/sampler.cpp:22`: `cudaMemcpyAsync` from pageable memory: it returns once the rows are staged, so `rows` can be freed right after; it is stream-ordered, so it cannot overwrite `rows_` while the previous step's kernel on the same stream still reads it. Using one `Sampler` from two streams would race. For pageable sources the host also waits for the stream first; harmless in `generate`, which syncs every step anyway, but M4's CUDA Graph path should stage rows in pinned memory.
- `src/runtime/sampler.cpp:24`: with zero rows `sample` still runs its shape checks and returns before launching.
- `src/runtime/generate.cpp:21`, `:28`: one-row `Sampler` for greedy too (one 32-byte upload per step), so the free-running HF test now exercises the new kernel's greedy branch.
- `src/app/generate_command.cpp`: parameters are validated before the tokenizer and the 2.2 GB weights load.
- Tests: the top-p tests assert that each setting is at least 1e-4 (1e-3 for the small row) away from a cut before the chi-square, so float summation order cannot flip a token in or out; `truncate()` in the test is the double-precision reference with the same tie rule. Survived mutants: `>` for `>=` in the top-k count (moves the threshold one float key down; equivalent unless a token sits exactly one float ulp below the k-th value) and masking with `s - 20` instead of `-inf` (a masked token almost never wins).

## Branch 19: sched/scheduler (Part A `Request` plus a Part C stub; nothing for line-by-line review)

- `Request` (`include/engine/sched/request.h`) is a plain aggregate so tests and the bench can use designated initializers; every field has a default member initializer because GCC's `-Wmissing-field-initializers` (from `-Wextra`) fires on designated initializers that skip a field without one, and the gate treats that as a failure.
- `max_new_tokens` lives on `Request`, not in `SamplingParams` as the build plan sketched: the sampler never reads it, and `SampleRow` (uploaded to the GPU every step) stays 24 bytes.
- `validate(request, max_context)` checks `prompt + max_new_tokens - 1 <= max_context`, the same rule as `generate`: the last sampled token is never fed, so it never needs a KV slot.
- The scheduler owns its `BlockManager` (constructed from `SchedulerConfig`), so nothing else can change block state behind its back, and the runner reads it through `blocks()`. Its private members are Joe's choice (as with `BlockManager`); the header declares only the public interface.
- `Step` lists sequence ids only; the runner gets everything else from `request(id)` (tokens via `token(pos)`, `SamplingParams`, `output.size()` as the sampling step) and `blocks()` (lengths, slots, tables). After `schedule()`, `blocks().length(s) == request(s).length()` for every sequence in the step, so a prefill feeds positions `[0, length)` and a decode feeds `length - 1`.
- Admission uses `can_allocate(L)`, not the build plan's `L + 1`: with the constructor's capacity check exactly tight (`num_blocks - 1 == ceil(max_context / block_size)`), a request re-admitted with one token left has `L = max_context`, and `L + 1` would need a block that can never exist, so it would wait forever. Mutant 4 (admit at `L + 1`) fails six tests.
- The liveness argument (in the spec): a running sequence never exceeds `max_context` tokens, the cache holds one such sequence alone, and `max_prefill_tokens >= max_context`; so the front of the queue always fits once nothing runs, and the earliest admitted sequence never has to preempt itself. Hence `schedule()` returns an empty step exactly when the scheduler is idle, which the random test asserts every step.
- Retirement happens in `update()`, not at the start of the next iteration as in the build plan's numbered list: the blocks come back one step earlier and `finished` is stamped when the last token arrives.
- The tests' fake model hashes the whole context read back from a fake cache of token ids through `blocks().table()`; a scheduler that re-prefills only the prompt, forgets to free, or hands out a slot twice makes the read-back context differ from the solo run.

## Branch 21: tools/bench (Part A; nothing for line-by-line review, two spots worth a look)

- `src/runtime/bench.cpp`: `first_token` and `finished` are stamped in the `generate` callback, which runs after `cudaStreamSynchronize` and the token's copy to the host, so they are host times of tokens that really exist. `arrived` is the scheduled arrival, not the start of service, so queueing counts in TTFT.
- `src/app/bench_command.cpp`: `decode_weight_bytes = stats.bytes - embed.bytes` unless `tie_embeddings` (a decode step reads one embedding row but every other weight once); bandwidth divides it by the TPOT p50 and ignores KV reads, as the build plan's metric does.
- Every request is validated against the runner's context before any runs, and both CSVs are opened before the weights load, so mistakes fail in about a second.
- Each commit passed the gate alone (GPU up): 284, 296, 302, 302 debug tests.
