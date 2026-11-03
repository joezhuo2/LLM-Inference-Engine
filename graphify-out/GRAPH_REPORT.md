# Graph Report - llm-inference-engine  (2026-11-03)

## Corpus Check
- 445 files · ~868,122 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 181 file(s) not represented in the graph (top: .csv 154, (none) 26, .css 1)

## Summary
- 4939 nodes · 10502 edges · 270 communities (210 shown, 60 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 1090 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bdd82d68`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)
- TEST
- tinyllama_advanced.py
- cuda_check.h
- ModelRunner
- results/story.tsx
- Review notes
- Buffer (Task 1)
- TEST
- bench_command.cpp
- .data
- TEST
- build-timeline.tsx
- TEST
- TEST
- TEST
- TEST
- vector
- pathlib
- MappedFile
- Request
- CudaStream
- from_bf16
- Shape
- TEST
- ModelWeights
- TEST
- KvCache
- PagedRunner
- TEST
- SafetensorsHeader
- Tensor
- [slug]/page.tsx
- Options
- BlockManager
- baseline_llamacpp.py
- visualizer.tsx
- Scheduler
- ModelConfig
- TEST
- write_kv
- RequestMetrics
- Summary
- TEST_P
- fail
- sync-data.mjs
- PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)
- StepBroadcaster
- GemmTest
- mapped_file.cpp
- react
- TextStreamTest
- block_manager.cpp
- TEST
- csv.cpp
- TEST_F
- WeightUploadTest
- Status
- compare_bf16
- Options
- gpu-floorplan.tsx
- gemm/story.tsx
- cuda-graphs/story.tsx
- make_workload
- WorkloadConfig
- Tokenizer
- README.md
- types.ts
- ReferencePrompt
- SchedulerConfig
- PinnedBuffer
- Batch
- Part 1 - Engine (M1 Foundations)
- expected_tensors
- SafetensorsFile
- SamplingParams
- AddWarpShuffle.cu
- vector
- parse_safetensors_header
- TEST_F
- Changelog
- bandwith_test.cu
- golden_dir
- rope/story.tsx
- paged-kv/story.tsx
- math
- TEST_F
- GpuBackend
- TEST
- TEST_P
- TEST
- Naive Oracle Kernels (Branch 11)
- Forward Pass (Branch 13)
- paged_attention_decode Implementation (kernel/paged-attention-impl)
- parse_length_range
- TEST_F
- ChatRequest
- ModelTest
- DType
- StepTrace
- Batched Sampling (Branch 18)
- Tokenizer (Branch 10)
- Weight Upload (Branch 6)
- Sampling (Branch 14)
- ncu
- useLoop
- PagedBatch
- line-chart.tsx
- chat_prompt
- Fit
- TEST
- Block Manager Implementation (kv/block-manager-impl)
- Naive Attention (Branch 12)
- Bench Harness (Branch 21)
- Tensor and Buffers (Branch 2)
- Paged KV Cache (Branch 16)
- Checksums (Branch 7)
- cuBLAS GEMM (Branch 3)
- pull_request_template.md
- smoke.cu
- argparse
- ServingLoop
- TEST
- Build Scaffolding (Branch 1)
- attention-v2/story.tsx
- Safetensors Loader Stub (Branch 5)
- serving_loop.cpp
- profile_report.py
- serve_step
- Shape
- TEST_F
- data.ts
- TEST_F
- sampling/story.tsx
- DashModel
- live.ts
- Label
- TEST
- TEST
- TEST
- baseline_vllm.py
- TEST
- attention_kernel
- LLM Inference Engine Setup Guide
- Profiling
- Frame
- scene-figures.tsx
- tensor.h
- Shape
- bandwith_test.md
- dram_throughput.md
- format.sh
- Batch
- Blocks
- TEST_F
- trace.ts
- add_rmsnorm_vec_kernel
- TEST
- as_bytes
- StopMatcher
- plot.py
- version
- swiglu_value
- RunGrouping
- Completion
- token-rain.tsx
- attention/story.tsx
- HttpServer::Impl
- decode_step_b16.md
- Plot
- feed.ts
- Safe Tensor Reader (Task 2)
- Results
- Scheduler (Branch 19)
- Attribution
- write_kv Implementation (kernel/write-kv-impl)
- usePrefersReducedMotion
- nvtx.h
- TEST
- reproduce.sh
- LlamaBench
- test_baseline_vllm.py
- decode_step.md
- decode_step_graphs.md
- rope_kernel
- read
- TEST
- TEST
- Docs (Branch 8)
- OwnedStream
- Paged Decode Attention (Branch 17)
- Shape
- setup_llamacpp.sh
- profile.sh
- .update
- Options
- PageTransition
- Repo Housekeeping (Branch 9)
- LLM Inference Engine Build Plan
- Pending
- ReplayPlayer
- cellNumber
- compilerOptions
- package.json
- MetricsWindow
- Technical notes
- app/layout.tsx
- TEST_F
- chat_api.cpp
- HttpOptions
- TEST
- MetricsSnapshot
- Web site plan (Part 2)
- Kernels section plan
- 1. Project context and decisions
- ChatPromptsTest
- run
- Core concepts you must understand
- Original prompt (verbatim)
- Working rules
- metrics_window.cpp
- TokenEvent
- dependencies
- LoadReport
- devDependencies
- Branch 73 `server/http` summary (issue #95, engine half)
- Live
- load_report.py
- scripts
- Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95)
- Testing, correctness, and benchmarking
- Finished
- Counters
- TEST
- load_chat_prompts
- Part 2: polish, UI/UX, and enhancements
- TextStream
- @number-flow/react
- LlamaBatchedBench
- row
- Branch 63 `web/polish` (web worktree), from origin/main 1f5d259 (#69 merged), not pushed
- Scoreboard
- Step 8: Day-1 smoke tests and troubleshooting
- Metrics
- Workload
- ProfileTest
- Web site
- KV Block Manager (Branch 15)
- stop_matcher.cpp
- Correctness
- load_sweep.sh
- setup_vllm.sh
- word_decode
- branch45_summary.md
- branch51_summary.md
- branch56_summary.md
- branch57_summary.md
- branch61_summary.md
- branch62_summary.md
- claude/README.md
- verify.sh
- AGENTS.md
- postcss.config.mjs

## God Nodes (most connected - your core abstractions)
1. `useLoop()` - 98 edges
2. `Part 1 - Engine (M1 Foundations)` - 90 edges
3. `Label()` - 89 edges
4. `Changelog` - 86 edges
5. `Frame()` - 75 edges
6. `Tensor` - 69 edges
7. `ServingLoop` - 51 edges
8. `Request` - 49 edges
9. `Scheduler` - 49 edges
10. `react` - 47 edges

## Surprising Connections (you probably didn't know these)
- `Reading these numbers` --references--> `ModelRunner`  [INFERRED]
  docs/results.md → include/engine/runtime/model_runner.h
- `Interface and errors` --references--> `table()`  [INFERRED]
  docs/exercises/learning_log.md → scripts/profile_report.py
- `Draft PR description` --references--> `step_json()`  [INFERRED]
  docs/claude/branch73_summary.md → src/bench/trace.cpp
- `Ownership` --references--> `embedding()`  [INFERRED]
  docs/claude/build_plan.md → src/kernels/embedding.cu
- `Decode bandwidth` --references--> `paged_attention_decode_v2()`  [INFERRED]
  docs/results.md → src/kernels/paged_attention_v2.cu

## Import Cycles
- None detected.

## Communities (270 total, 60 thin omitted)

### Community 0 - "Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)"
Cohesion: 0.09
Nodes (26): tools/checksum, Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge), dtype_name(), TensorChecksum, abs_sum, dtype, name, shape (+18 more)

### Community 1 - "TEST"
Cohesion: 0.11
Nodes (3): check(), run_paged(), TEST()

### Community 2 - "tinyllama_advanced.py"
Cohesion: 0.13
Nodes (21): capture(), keep(), chat_ids(), dump_prompt(), main(), apply_rope(), forward(), generate() (+13 more)

### Community 3 - "cuda_check.h"
Cohesion: 0.07
Nodes (5): Event, event, TEST(), TEST(), TEST()

### Community 4 - "ModelRunner"
Cohesion: 0.06
Nodes (27): runtime/wire-fast-kernels, tools/bench, Branch 20 summary: model/paged-runner (Part B), Branch 21: tools/bench, Decisions (confirmed by Joe on Oct 11), Measured (RTX 5060, release, Oct 11), Usage, Worth a look (+19 more)

### Community 5 - "results/story.tsx"
Cohesion: 0.05
Nodes (67): Row-major GEMM through cuBLAS, Legend(), Stats(), GroupLegend, batchAxis, benchStats(), contextAxis, contextY (+59 more)

### Community 6 - "Review notes"
Cohesion: 0.12
Nodes (18): Branch 13: model/forward (Part B wiring, greedy and generate are line-by-line review items; TextStream and the CLI are Part A), Branch 14: sampling/temperature (Part B kernel and generate are line-by-line review items; the CLI flags are Part A), Branch 17: kernel/paged-decode-attention (Part C stub; nothing for line-by-line review), Branch 2: core/tensor-and-buffers, Branch 3: kernel/cublas-gemm (Part B, line-by-line review item), Branch 4: model/config, Branch 5: loader/safetensors-stub, Review notes (+10 more)

### Community 7 - "Buffer (Task 1)"
Cohesion: 0.10
Nodes (20): Atomics and shared memory, Bandwith Test (Task 3/CUDA Task 1), Buffer (Task 1), Core C++, Core Ideas, Core Ideas, Core Ideas, Execution model (+12 more)

### Community 10 - "bench_command.cpp"
Cohesion: 0.11
Nodes (3): Scheduler::cancel(), Scheduler::update(), gpu_name()

### Community 11 - ".data"
Cohesion: 0.08
Nodes (13): DeviceBuffer, ptr_, DeviceBuffer::DeviceBuffer(), TEST(), TEST(), download(), TEST_F(), upload() (+5 more)

### Community 13 - "build-timeline.tsx"
Cohesion: 0.06
Nodes (50): web/polish, cards, Overview(), BuildTimeline(), captions, ease, entries, EntryView() (+42 more)

### Community 14 - "TEST"
Cohesion: 0.11
Nodes (11): Branch 16: kv/paged-cache (Part A plus a Part C stub; nothing for line-by-line review), Case, block_size, head_dim, heads, kv_heads, num_blocks, check() (+3 more)

### Community 15 - "TEST"
Cohesion: 0.05
Nodes (32): CopyOp, bytes, dst, name, shape, src, LayerLayout, ln1 (+24 more)

### Community 16 - "TEST"
Cohesion: 0.09
Nodes (3): make(), sixteen_requests(), TEST()

### Community 17 - "TEST"
Cohesion: 0.10
Nodes (18): attention_scale(), AttentionLayoutTest, Cache, l_, rows_, cpu_attention(), Expected, out (+10 more)

### Community 19 - "pathlib"
Cohesion: 0.09
Nodes (14): engine_command(), llama_command(), llama_ttfts(), main(), measure(), read_engine_ttft(), read_rows(), write_rows() (+6 more)

### Community 20 - "MappedFile"
Cohesion: 0.11
Nodes (7): MappedFile, data_, size_, TEST(), TempFile, path_, TEST()

### Community 21 - "Request"
Cohesion: 0.07
Nodes (24): Request, arrived, finished, first_token, id, max_new_tokens, output, params (+16 more)

### Community 22 - "CudaStream"
Cohesion: 0.13
Nodes (5): GenerateTest, stream, TEST_F(), CudaStream, stream_

### Community 23 - "from_bf16"
Cohesion: 0.06
Nodes (24): cpu_add(), AddRmsnormShapeTest, cpu_add(), cpu_rmsnorm(), Result, norm, sum, run() (+16 more)

### Community 24 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, hidden, shift, tokens

### Community 26 - "ModelWeights"
Cohesion: 0.06
Nodes (27): loader/weight-upload, Branch 6: loader/weight-upload (the upload is a line-by-line review item), LayerWeights, ln1, ln2, wdown, wgu, wo (+19 more)

### Community 27 - "TEST"
Cohesion: 0.10
Nodes (9): fail(), field(), load_config(), parse_config(), require_if_present(), expect_rejected(), TEST(), with() (+1 more)

### Community 28 - "KvCache"
Cohesion: 0.14
Nodes (7): KvCache, buffer_, head_dim_, kv_heads_, small(), TEST(), tinyllama()

### Community 29 - "PagedRunner"
Cohesion: 0.07
Nodes (25): model/forward, model/paged-runner, Stubs (Part C), CUevent_st, PagedRunner, act_, attn_, blas_ (+17 more)

### Community 30 - "TEST"
Cohesion: 0.15
Nodes (4): Buffer, memory_address, requested_size, TEST()

### Community 31 - "SafetensorsHeader"
Cohesion: 0.14
Nodes (16): loader/safetensors-stub, SafetensorsHeader, data_offset, tensors, TensorEntry, bytes, dtype, offset (+8 more)

### Community 32 - "Tensor"
Cohesion: 0.07
Nodes (24): Pages, Tensor, data, dtype, kMaxDims, ndim, shape, main() (+16 more)

### Community 33 - "[slug]/page.tsx"
Cohesion: 0.06
Nodes (45): web/kernels-data, 53 web/kernels-data (2 commits), 54 web/kernels-template (4 commits), Branches 53 `web/kernels-data` and 54 `web/kernels-template` (web worktree), stacked on web/overview (merged as #60, so a PR against main shows only these 6 commits), not pushed, next, shiki, config, DashboardPage() (+37 more)

### Community 34 - "Options"
Cohesion: 0.11
Nodes (19): Options, graphs, max_batch, model, num_requests, out, output_len, profile (+11 more)

### Community 35 - "BlockManager"
Cohesion: 0.06
Nodes (24): kv/block-manager, kv/block-manager-impl, sched/scheduler-impl, web/kernel-paged-kv, C7. Continuous batching scheduler (week 5), Admission and preemption, BlockManager, block_size_ (+16 more)

### Community 36 - "baseline_llamacpp.py"
Cohesion: 0.30
Nodes (13): batched_command(), batched_main(), bench_main(), gpu_name(), main(), parse_batched_bench(), parse_llama_bench(), run_llama_bench() (+5 more)

### Community 37 - "visualizer.tsx"
Cohesion: 0.07
Nodes (31): Cell, Mode, Pool, Seq, Shape, BlockDetails(), Colors, corners() (+23 more)

### Community 38 - "Scheduler"
Cohesion: 0.11
Nodes (23): sched/scheduler, Decisions in branch 19 (Claude's, open to Joe's change), Done but not merged, Scheduler, block_manager_, config_, finished_list_, pending_step_ (+15 more)

### Community 39 - "ModelConfig"
Cohesion: 0.06
Nodes (32): model/config, Branches, How Joe verifies a branch, M1 plan, What each branch contains, Decisions in branch 16 (Claude's, open to Joe's change), Config semantics, Model Config (Branch 4) (+24 more)

### Community 40 - "TEST"
Cohesion: 0.14
Nodes (3): bf16_bytes(), bytes_of(), TEST()

### Community 41 - "write_kv"
Cohesion: 0.07
Nodes (30): docs/write-kv-profile, kernel/write-kv-impl, sampling/temperature, tools/profiling, tools/sample-bench, C3. Kernels (weeks 2 to 6), Branches (order confirmed by Joe on Oct 10), Decisions in branch 15 (Claude's, open to Joe's change) (+22 more)

### Community 42 - "RequestMetrics"
Cohesion: 0.14
Nodes (12): RequestMetrics, arrival_s, id, latency_s, output_tokens, preemptions, prompt_tokens, tpot_s (+4 more)

### Community 43 - "Summary"
Cohesion: 0.13
Nodes (15): Stat, mean, p50, p90, p99, Summary, latency_s, output_throughput (+7 more)

### Community 44 - "TEST_P"
Cohesion: 0.19
Nodes (9): cpu_rope(), gpu_rope(), Layout, head_dim, heads, kv_heads, RopeLayoutTest, spread_positions() (+1 more)

### Community 45 - "fail"
Cohesion: 0.22
Nodes (5): fail(), Tokenizer::append_text(), Tokenizer::decode(), Tokenizer::encode(), Tokenizer::Tokenizer()

### Community 46 - "sync-data.mjs"
Cohesion: 0.06
Nodes (35): web/kernel-cuda-graphs, Branch 59 `web/kernel-cuda-graphs` (web worktree), from origin/main 03fefac (#65 merged), not pushed, replayKv(), meta, requests, steps, cells(), isSeparator() (+27 more)

### Community 47 - "PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)"
Cohesion: 0.10
Nodes (19): Calling cuBLAS with row-major weights, Model and system architecture, Model config ([config.json](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json)), Repository layout, System architecture, Tokenizer, Weight tensors (safetensors names and shapes), Attention (+11 more)

### Community 48 - "StepBroadcaster"
Cohesion: 0.05
Nodes (14): StepBroadcaster, capacity_, closed_, mutex_, subscribers_, Channel, capacity_, items_ (+6 more)

### Community 49 - "GemmTest"
Cohesion: 0.11
Nodes (9): Blas, cublasContext, GemmShapeTest, GemmTest, blas_, stream_, workspace_, TEST() (+1 more)

### Community 50 - "mapped_file.cpp"
Cohesion: 0.09
Nodes (9): main(), print_version(), fail(), Fd, fd, MappedFile::MappedFile(), SafetensorReader, main() (+1 more)

### Community 51 - "react"
Cohesion: 0.09
Nodes (28): clsx, react, Code(), ConnectBar(), ConnectHelp(), DEFAULT_URL, LiveBar(), noSubscribe() (+20 more)

### Community 52 - "TextStreamTest"
Cohesion: 0.18
Nodes (3): TEST_F(), TextStreamTest, tok

### Community 53 - "block_manager.cpp"
Cohesion: 0.19
Nodes (4): BlockManager::can_append(), BlockManager::free(), BlockManager::length(), BlockManager::slot()

### Community 55 - "csv.cpp"
Cohesion: 0.26
Nodes (5): csv_escape(), csv_number(), summary_fields(), write_csv_row(), write_request_csv()

### Community 56 - "TEST_F"
Cohesion: 0.18
Nodes (3): BenchTest, stream, TEST_F()

### Community 57 - "WeightUploadTest"
Cohesion: 0.17
Nodes (5): TEST_P(), WeightUploadTest, file_, layout_, stream_

### Community 58 - "Status"
Cohesion: 0.07
Nodes (28): kernel/paged-attention-impl, kernel/paged-decode-attention, Decisions in branch 20 (Claude's, open to Joe's change), M1 verification (Joe's run, Oct 9, in the Claude worktree), Milestones, Next steps for Claude, Part C stubs that exist, Snapshot 2026-10-26 (after midnight): CUDA Graphs page (+20 more)

### Community 59 - "compare_bf16"
Cohesion: 0.06
Nodes (18): kernel/add-rmsnorm-impl, kernel/naive-oracles, Branch 11: kernel/naive-oracles (Part B, every line is a review item), rmsnorm(), TEST(), cpu_rmsnorm(), gpu_rmsnorm(), RmsnormShapeTest (+10 more)

### Community 60 - "Options"
Cohesion: 0.17
Nodes (9): Options, max_new_tokens, model, prompt, seed, system, temperature, top_k (+1 more)

### Community 61 - "gpu-floorplan.tsx"
Cohesion: 0.08
Nodes (41): A, B, bandwidth, blockTable, Box, bx(), C, checksum (+33 more)

### Community 62 - "gemm/story.tsx"
Cohesion: 0.09
Nodes (36): sharedPoints, bytesOf(), GemvFigure(), mb(), Projection, ProjectionsFigure(), ReuseFigure(), RoofBarsFigure() (+28 more)

### Community 63 - "cuda-graphs/story.tsx"
Cohesion: 0.08
Nodes (35): buckets, captured, CaptureFigure(), eagerLayer, EagerTimelineFigure(), fields, fmt(), GapFigure() (+27 more)

### Community 64 - "make_workload"
Cohesion: 0.21
Nodes (7): LengthRange, max, min, check(), make_workload(), SplitMix64, state_

### Community 65 - "WorkloadConfig"
Cohesion: 0.12
Nodes (16): bench/chat-prompts, Workload, arrivals, requests, WorkloadConfig, bos_id, eos_id, num_requests (+8 more)

### Community 66 - "Tokenizer"
Cohesion: 0.14
Nodes (8): SentencePieceProcessor, Tokenizer, sentinel_, sp_, unk_id_, ServerTest, tok, TEST()

### Community 67 - "README.md"
Cohesion: 0.11
Nodes (17): Branch 44 `tools/llamacpp-baseline` (M5, Part A), C9. CLI and benchmark harness (weeks 3 and 7), Architecture, Beyond the MVP, Building, Design decisions, Goals, Hardware (+9 more)

### Community 68 - "types.ts"
Cohesion: 0.09
Nodes (27): addRmsnorm, oneToken, at2048, attentionRow, attentionV2, cudaGraphs, eager, first() (+19 more)

### Community 69 - "ReferencePrompt"
Cohesion: 0.16
Nodes (14): tokenizer/sentencepiece, Reference format (`tests/golden/reference/`, generated, gitignored), load_reference_manifest(), reference_dir(), reference_skip_reason(), ReferencePrompt, file, generated_ids (+6 more)

### Community 70 - "SchedulerConfig"
Cohesion: 0.10
Nodes (16): SchedulerConfig, block_size, eos_id, max_batch, max_context, max_prefill_tokens, num_blocks, Scheduler::Scheduler() (+8 more)

### Community 71 - "PinnedBuffer"
Cohesion: 0.22
Nodes (4): PinnedBuffer, ptr_, PinnedBuffer::PinnedBuffer(), TEST()

### Community 72 - "Batch"
Cohesion: 0.36
Nodes (4): Batch, sampler_, stream_, vocab_

### Community 73 - "Part 1 - Engine (M1 Foundations)"
Cohesion: 0.03
Nodes (64): Add RMSNorm Tuning (Branch 39), Attention v2 Page (Branch 67), Attention v2 README (Branch 78), Baselines (Branch 29), Batched Benchmarking (Branch 23), Cancel and Host Check (Branch 76), Chat Prompt Workloads (Branch 49), CI Hardening (build/ci-timeouts) (+56 more)

### Community 74 - "expected_tensors"
Cohesion: 0.20
Nodes (5): Expected, dtype, shape, expected_tensors(), TEST()

### Community 75 - "SafetensorsFile"
Cohesion: 0.10
Nodes (7): TEST(), cpu_greedy(), run_greedy(), TEST(), SafetensorsFile, file_, header_

### Community 76 - "SamplingParams"
Cohesion: 0.22
Nodes (8): SamplingParams, seed, temperature, top_k, top_p, validate(), drawn(), histogram()

### Community 78 - "vector"
Cohesion: 0.22
Nodes (9): Branch 18: sampling/batched (Part B kernel, Sampler and generate are line-by-line review items; the CLI flags are Part A), bf16_row(), repeat_rows(), softmax(), truncate(), Truncated, margin, p (+1 more)

### Community 79 - "parse_safetensors_header"
Cohesion: 0.28
Nodes (5): loader/safetensors-parser, get_dtype_size(), parse_safetensors_header(), throw_error(), throw_tensor_error()

### Community 80 - "TEST_F"
Cohesion: 0.14
Nodes (3): TEST_F(), TokenizerTest, tok

### Community 81 - "Changelog"
Cohesion: 0.05
Nodes (36): bench/prefill-ttft, bench/qps-sweep, bench/vllm-baseline, build/ci-timeouts, build/repo-housekeeping, Changelog, docs/attention-v2-readme, docs/graphs-gap-reading (+28 more)

### Community 82 - "bandwith_test.cu"
Cohesion: 0.36
Nodes (6): launch(), main(), timeKernel(), upload(), vectorAdd(), verify()

### Community 83 - "golden_dir"
Cohesion: 0.40
Nodes (3): golden_checksums_path(), golden_dir(), model_dir()

### Community 84 - "rope/story.tsx"
Cohesion: 0.08
Nodes (34): web/kernels-template, Branch 74 `web/results` (issue #92), summary, Commits, Draft PR description, Not verified, Open questions for Joe, Verified, What was built (+26 more)

### Community 85 - "paged-kv/story.tsx"
Cohesion: 0.11
Nodes (34): motion, Props, windowTimes(), AppendFigure(), first, fmt(), fullContext, heldAt() (+26 more)

### Community 86 - "math"
Cohesion: 0.17
Nodes (15): greedy_batch(), main(), run(), summary_from_batched_row(), bandwidth_fields(), _number(), percentile(), request_metrics() (+7 more)

### Community 87 - "TEST_F"
Cohesion: 0.09
Nodes (16): Arrival, at_iteration, id, max_new_tokens, prompt_len, bucket_of(), prompt_of(), Rig (+8 more)

### Community 88 - "GpuBackend"
Cohesion: 0.06
Nodes (22): server/http, HTTP Server (Branch 73), Backend, StepResult, finished, trace, GpuBackendOptions, graphs (+14 more)

### Community 90 - "TEST_P"
Cohesion: 0.07
Nodes (5): check(), PagedLayoutSplitTest, PagedSplitTest, run_paged(), TEST_P()

### Community 92 - "Naive Oracle Kernels (Branch 11)"
Cohesion: 0.29
Nodes (7): Embedding, Naive Oracle Kernels (Branch 11), Residual add, RMSNorm, RoPE kernel, SwiGLU kernel, Testing kernels

### Community 93 - "Forward Pass (Branch 13)"
Cohesion: 0.29
Nodes (7): engine generate, Forward Pass (Branch 13), Free-running generation, Golden harness, Greedy decoding, Model runner, Streaming text

### Community 94 - "paged_attention_decode Implementation (kernel/paged-attention-impl)"
Cohesion: 0.29
Nodes (7): Host side, Kernel structure, Numerics, paged_attention_decode Implementation (kernel/paged-attention-impl), Shared memory, Softmax and reductions, Testing

### Community 96 - "TEST_F"
Cohesion: 0.07
Nodes (12): byte_encode(), chat_body(), HttpServerTest, backend, engine, http, listener, loop (+4 more)

### Community 97 - "ChatRequest"
Cohesion: 0.11
Nodes (11): ApiError, ChatRequest, max_tokens, messages, params, seed, stop, stream (+3 more)

### Community 99 - "DType"
Cohesion: 0.07
Nodes (22): core/tensor-and-buffers, Cadence, Git workflow, Interfaces to agree on in week 0, Ownership, Team split and workflow, CUstream_st, DType (+14 more)

### Community 100 - "StepTrace"
Cohesion: 0.07
Nodes (16): StepTrace, end, finished, free_blocks, prefill, running, seqs, start (+8 more)

### Community 101 - "Batched Sampling (Branch 18)"
Cohesion: 0.33
Nodes (6): Batched Sampling (Branch 18), Command line, Per-row sampling, Sampling parameters, Top-k by threshold search, Top-p by threshold search

### Community 102 - "Tokenizer (Branch 10)"
Cohesion: 0.33
Nodes (6): Chat template, Dependencies, Golden tests for the tokenizer, Reference dumps, Tokenization, Tokenizer (Branch 10)

### Community 103 - "Weight Upload (Branch 6)"
Cohesion: 0.33
Nodes (6): Double-buffered upload, Loading end to end, Streams and ordering, Testing, Weight layout, Weight Upload (Branch 6)

### Community 104 - "Sampling (Branch 14)"
Cohesion: 0.33
Nodes (6): M2 criterion, Sampled generation, Sampling (Branch 14), Sampling flags, Shared argmax, Temperature sampling

### Community 105 - "ncu"
Cohesion: 0.09
Nodes (23): Cloud GPUs (only for final benchmarks), Hardware and toolchain, Learning path (both of you, weeks 0 to 2), Operating system setup, Risks, Risks, mitigations, and learning resources, Sources, Toolchain (pin these versions in the README) (+15 more)

### Community 106 - "useLoop"
Cohesion: 0.11
Nodes (33): useLoop(), CutFigure(), GraphKey, GraphsFigure(), MergeFigure(), Occupancy, OccupancyFigure(), partials (+25 more)

### Community 107 - "PagedBatch"
Cohesion: 0.06
Nodes (26): kernel/m4-stubs, runtime/graphs-split-attention, runtime/wire-attention-split, Branch 22 `kernel/m4-stubs` (M4 Part C stubs for Joe), Decisions I made (tell me if you disagree), Not done on purpose, Validation, What Joe implements (all todo_joe, run `ctest --preset debug -L todo_joe`) (+18 more)

### Community 108 - "line-chart.tsx"
Cohesion: 0.09
Nodes (31): Axis, LineChart(), pad, pathOf(), Props, Readout(), Row, scale() (+23 more)

### Community 109 - "chat_prompt"
Cohesion: 0.14
Nodes (9): Branches, Decisions Joe confirmed on Oct 9 ("yes to all"), M2 plan: correct naive engine (due Oct 18), Open questions for later in M2, Results (branch 13, Oct 9), Branch 10: tokenizer/sentencepiece (Part A), add_generate_command(), run() (+1 more)

### Community 110 - "Fit"
Cohesion: 0.33
Nodes (6): chi_square_critical(), Fit, chi2, critical, impossible, goodness_of_fit()

### Community 111 - "TEST"
Cohesion: 0.09
Nodes (3): check(), run_paged(), TEST()

### Community 112 - "Block Manager Implementation (kv/block-manager-impl)"
Cohesion: 0.40
Nodes (5): Allocation, Block Manager Implementation (kv/block-manager-impl), Construction, Errors, Slots

### Community 113 - "Naive Attention (Branch 12)"
Cohesion: 0.40
Nodes (5): Attention against HF, CI container images, Contiguous KV cache, Naive Attention (Branch 12), Naive attention kernel

### Community 114 - "Bench Harness (Branch 21)"
Cohesion: 0.40
Nodes (5): Bench Harness (Branch 21), Metrics and CSV, Running a workload, Synthetic workloads, The bench command

### Community 115 - "Tensor and Buffers (Branch 2)"
Cohesion: 0.40
Nodes (5): C++, CUDA error handling, Memory, Tensor and Buffers (Branch 2), Testing

### Community 116 - "Paged KV Cache (Branch 16)"
Cohesion: 0.40
Nodes (5): Cache sizing, KV cache memory, Paged KV Cache (Branch 16), Testing kernels, write_kv

### Community 117 - "Checksums (Branch 7)"
Cohesion: 0.40
Nodes (5): Checksum math, Checksums (Branch 7), CLI, Golden tests, Python reference

### Community 118 - "cuBLAS GEMM (Branch 3)"
Cohesion: 0.50
Nodes (4): cuBLAS GEMM (Branch 3), cuBLAS setup, Numerics, Testing

### Community 119 - "pull_request_template.md"
Cohesion: 0.40
Nodes (4): How it was tested, Performance (if relevant), Screenshots (ncu / nsys, if relevant), What changed

### Community 121 - "argparse"
Cohesion: 0.14
Nodes (18): checksums(), main(), main(), gemm_cost(), main(), markdown(), model_shape(), op_costs() (+10 more)

### Community 122 - "ServingLoop"
Cohesion: 0.07
Nodes (15): ServingLoop, cancels_, counters_, decode_, eos_id_, inbox_mutex_, inbox_ready_, live_ (+7 more)

### Community 123 - "TEST"
Cohesion: 0.07
Nodes (14): check(), consecutive(), Layout, head_dim, heads, kv_heads, random_positions(), TEST() (+6 more)

### Community 124 - "Build Scaffolding (Branch 1)"
Cohesion: 0.50
Nodes (4): Build Scaffolding (Branch 1), CI and tooling, CMake and dependencies, Python

### Community 125 - "attention-v2/story.tsx"
Cohesion: 0.10
Nodes (28): web/kernel-add-rmsnorm, Branch 55 `web/kernel-add-rmsnorm` (web worktree), from origin/main c3e86c3 (#61 merged), not pushed, chunkMaxes, chunkScores, LanesFigure(), OnlineSoftmaxFigure(), Pair, PairBars() (+20 more)

### Community 126 - "Safetensors Loader Stub (Branch 5)"
Cohesion: 0.50
Nodes (4): Mapping files, Safetensors Loader Stub (Branch 5), Safetensors parsing (your Part C), Testing

### Community 127 - "serving_loop.cpp"
Cohesion: 0.11
Nodes (13): request_metrics(), seconds(), ServingLoop::cancel(), ServingLoop::deliver(), ServingLoop::drop(), ServingLoop::finish(), ServingLoop::metrics(), ServingLoop::record_step() (+5 more)

### Community 128 - "profile_report.py"
Cohesion: 0.14
Nodes (21): Branch 15: kv/block-manager (Part C stub; nothing for line-by-line review), grouped(), plot_ttft(), table_markdown(), attribute(), bandwidth_section(), busy_and_span(), export_sqlite() (+13 more)

### Community 129 - "serve_step"
Cohesion: 0.11
Nodes (20): bench/batched, docs/api-contract, runtime/cuda-graphs-impl, sampling/batched, Branch 48 `docs/api-contract` (PR summary), Open items and notes, Top-k and top-p (branch 18, measured Oct 10), DecodeGraphs (+12 more)

### Community 130 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, b, in, out

### Community 131 - "TEST_F"
Cohesion: 0.10
Nodes (6): TEST(), to_host(), ServeTest, prompts, stream, TEST_F()

### Community 132 - "data.ts"
Cohesion: 0.08
Nodes (21): metadata, Results(), measured(), benchRun, batches, Doc, docPath, format() (+13 more)

### Community 133 - "TEST_F"
Cohesion: 0.13
Nodes (5): PagedRunnerTest, stream, prompt_of(), TEST_F(), to_host()

### Community 134 - "sampling/story.tsx"
Cohesion: 0.12
Nodes (29): ArgmaxFigure(), Bars(), CostFigure(), Fp64Figure(), gumbel(), GumbelFigure(), keys, logits (+21 more)

### Community 136 - "live.ts"
Cohesion: 0.13
Nodes (12): Blocked, getJson(), HttpError, isLoopback(), isPrivateHost(), LiveConnection, Metrics, metricsTiles() (+4 more)

### Community 137 - "Label"
Cohesion: 0.13
Nodes (27): FirstFusedFigure(), fusedKb64, JobFigure(), lanesStart, MeasuredBars(), naiveKb64, Packet(), phases (+19 more)

### Community 140 - "TEST"
Cohesion: 0.09
Nodes (9): CorsPolicy, origins_, CorsPolicy::allows(), CorsPolicy::CorsPolicy(), CorsPolicy::preflight_headers(), CorsPolicy::response_headers(), validate_origin(), header() (+1 more)

### Community 141 - "baseline_vllm.py"
Cohesion: 0.15
Nodes (18): client_stats(), completion_payload(), drive(), fetch(), gpu_name(), main(), parse_preemptions(), parse_rates() (+10 more)

### Community 144 - "LLM Inference Engine Setup Guide"
Cohesion: 0.07
Nodes (27): 1.1 Update Windows and the NVIDIA driver, 1.2 Install WSL2 and Ubuntu 24.04, 1.3 Configure WSL2 resources and networking, 1.4 Install the editor and terminal, 1.5 Rules for working in WSL2, 2.1 Base packages, 2.2 CUDA toolkit (WSL-Ubuntu repo), 2.3 Environment variables (+19 more)

### Community 145 - "Profiling"
Cohesion: 0.05
Nodes (36): kernel/paged-attention-split-impl, kernel/paged-attention-split-stub, kernel/paged-attention-v2-impl, runtime/wire-attention-v2, runtime/wire-attention-v2, Attention shared memory (Branch 67), Decode CUDA Graphs (Branch 34), Fast kernels in the forward pass (Branch 36) (+28 more)

### Community 146 - "Frame"
Cohesion: 0.14
Nodes (25): Frame(), bf16(), ChainFigure(), GridFigure(), peakGbs, silu(), SmFigure(), TimeBars() (+17 more)

### Community 147 - "scene-figures.tsx"
Cohesion: 0.11
Nodes (25): web/kernel-gemm, Branch 58 `web/kernel-gemm` (web worktree), from origin/main 7e8df6b (#64 merged), not pushed, metric(), batch1, batch16, decode, GenerateFigure(), logits (+17 more)

### Community 148 - "tensor.h"
Cohesion: 0.04
Nodes (16): scale_kernel(), add_kernel(), embedding_kernel(), Blas::Blas(), paged_attention_split_merge_kernel(), paged_attention_split_partial_kernel(), paged_attention_decode_v2_kernel(), rmsnorm_kernel() (+8 more)

### Community 149 - "Shape"
Cohesion: 0.67
Nodes (3): Shape, hidden, tokens

### Community 154 - "Batch"
Cohesion: 0.09
Nodes (32): kernel/paged-attention-v2-stub, PagedLayoutTest, TEST_P(), PagedLayoutV2Test, TEST_P(), Batch, k, l (+24 more)

### Community 156 - "TEST_F"
Cohesion: 0.16
Nodes (3): BatchedBenchTest, stream, TEST_F()

### Community 157 - "trace.ts"
Cohesion: 0.13
Nodes (21): `POST /v1/chat/completions`, model(), path, ModelOptions, RESERVED, Seq, UNKNOWN, emptyTiles (+13 more)

### Community 158 - "add_rmsnorm_vec_kernel"
Cohesion: 0.31
Nodes (6): kernel/add-rmsnorm-tune, add_rmsnorm_kernel(), add_rmsnorm_vec_kernel(), blockAllReduceSum(), blockReduceSum(), warpReduceSum()

### Community 159 - "TEST"
Cohesion: 0.11
Nodes (7): TEST(), GuardedBf16, begin_, buffer_, kGuard, kSentinel, n_

### Community 160 - "as_bytes"
Cohesion: 0.27
Nodes (5): as_bytes(), safetensors_file(), expect_rejected(), expect_rejected_header(), parse()

### Community 161 - "StopMatcher"
Cohesion: 0.10
Nodes (9): Result, stopped, text, StopMatcher, held_, stops_, StopMatcher::push(), run() (+1 more)

### Community 162 - "plot.py"
Cohesion: 0.16
Nodes (22): docs/results, tools/llamacpp-baseline, best(), environment_block(), main(), median_of(), parse_correctness(), parse_environment() (+14 more)

### Community 163 - "version"
Cohesion: 0.11
Nodes (15): build/m1-scaffolding, Baselines and plots, HuggingFace transformers, llama.cpp, Plots, Reading the comparison fairly, The workload, vLLM (+7 more)

### Community 164 - "swiglu_value"
Cohesion: 0.27
Nodes (6): kernel/swiglu-small-batch, kernel/swiglu-vec-impl, swiglu_kernel(), swiglu_value(), swiglu_vec_kernel(), SwigluShapeTest

### Community 166 - "Completion"
Cohesion: 0.12
Nodes (21): Completion, created, model, seq, FinishReason, Length, Stop, chunk() (+13 more)

### Community 167 - "token-rain.tsx"
Cohesion: 0.12
Nodes (19): web/overview, Branch 52 `web/overview` (worktree ~/code/llm-inference-engine-web), 4 commits STACKED on web/prototypes (1faf8ab), not pushed, useReplay(), IntroLoader(), kvBlocks, randomHex(), stages, createRain() (+11 more)

### Community 168 - "attention/story.tsx"
Cohesion: 0.12
Nodes (21): BlocksFigure(), others, OverviewFigure(), probs, scores, SmGrid(), SplitFigure(), table (+13 more)

### Community 169 - "HttpServer::Impl"
Cohesion: 0.19
Nodes (11): error_json(), HttpServer::HttpServer(), HttpServer::Impl, cors, encode, http, options, random_seed() (+3 more)

### Community 170 - "decode_step_b16.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 172 - "feed.ts"
Cohesion: 0.16
Nodes (19): Colors, readColors(), slotColor(), SLOTS, LaneRefs, Requests(), status(), tail() (+11 more)

### Community 173 - "Safe Tensor Reader (Task 2)"
Cohesion: 0.25
Nodes (8): Build, Bytes and types, Core Ideas, Errors and program entry, Memory mapping, Opening and inspecting files, Safe Tensor Reader (Task 2), Safetensors format

### Community 174 - "Results"
Cohesion: 0.11
Nodes (18): Batch-1 decode bandwidth, Decode bandwidth, Decode bandwidth, Environment, GPU time per op in a decode step, GPU time per op in a decode step, Launch gaps per forward step, Launch gaps per forward step (+10 more)

### Community 175 - "Scheduler (Branch 19)"
Cohesion: 0.50
Nodes (4): Continuous batching, Requests, Scheduler (Branch 19), Testing the scheduler

### Community 177 - "write_kv Implementation (kernel/write-kv-impl)"
Cohesion: 0.50
Nodes (4): Kernel indexing, Paged cache writes, Testing, write_kv Implementation (kernel/write-kv-impl)

### Community 178 - "usePrefersReducedMotion"
Cohesion: 0.17
Nodes (18): BlockDetails(), Cell, fmt(), Geo, geometry(), Hover, KvPool(), poolWindow() (+10 more)

### Community 180 - "TEST"
Cohesion: 0.10
Nodes (3): parse(), status_of(), TEST()

### Community 181 - "reproduce.sh"
Cohesion: 0.73
Nodes (5): bench(), graph_bench(), quiet(), reproduce.sh script, usage()

### Community 183 - "test_baseline_vllm.py"
Cohesion: 0.12
Nodes (8): event(), Options, ParseSse, Payload, request(), ServerLog, Stats, TokenCounting

### Community 184 - "decode_step.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 185 - "decode_step_graphs.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 186 - "rope_kernel"
Cohesion: 0.40
Nodes (3): InvFreq, v, rope_kernel()

### Community 188 - "TEST"
Cohesion: 0.11
Nodes (6): ScriptedBackend, delay_, next_, scheduler_, run_until_idle(), TEST()

### Community 189 - "TEST"
Cohesion: 0.10
Nodes (5): make(), next_token(), run_step(), solo(), TEST()

### Community 190 - "Docs (Branch 8)"
Cohesion: 0.50
Nodes (4): Docs (Branch 8), Ownership, Project overview, README

### Community 192 - "Paged Decode Attention (Branch 17)"
Cohesion: 0.67
Nodes (3): Paged attention, Paged Decode Attention (Branch 17), Testing attention

### Community 193 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, ff, shift, tokens

### Community 196 - ".update"
Cohesion: 0.12
Nodes (18): Conventions, CORS, Engine API contract, `GET /health`, `GET /v1/steps` (Server-Sent Events), Host header, HTTP endpoints (`engine serve`), Step object (+10 more)

### Community 198 - "Options"
Cohesion: 0.14
Nodes (16): add_serve_command(), allowed_hosts(), http_options(), is_loopback(), joined(), model_name(), Options, allow_hosts (+8 more)

### Community 199 - "PageTransition"
Cohesion: 0.19
Nodes (14): Loading(), Kernels(), metadata, Loading(), KernelCards(), LayerMap(), direction, PageTransition() (+6 more)

### Community 202 - "LLM Inference Engine Build Plan"
Cohesion: 0.10
Nodes (19): C1. Core infrastructure (week 1), C2. Weight loader (week 1), C4. KV block manager (week 4), C5. Model runner (weeks 2 to 5), C6. CUDA Graphs (week 6), C8. Sampler (weeks 3 and 5), If you fall behind, LLM Inference Engine Build Plan (+11 more)

### Community 203 - "Pending"
Cohesion: 0.12
Nodes (16): Pending, arrived, events, id, submission, Submission, max_tokens, params (+8 more)

### Community 205 - "cellNumber"
Cohesion: 0.15
Nodes (17): stepOps, share, End, ends, GatherFigure(), HeadFigure(), ids, LayerEnd() (+9 more)

### Community 206 - "compilerOptions"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+10 more)

### Community 207 - "package.json"
Cohesion: 0.11
Nodes (16): eslint, eslint-config-next, @fontsource/jetbrains-mono, liveline, react-dom, recharts, tailwindcss, @tailwindcss/postcss (+8 more)

### Community 208 - "MetricsWindow"
Cohesion: 0.16
Nodes (5): `GET /metrics`, MetricsWindow, finished_, tokens_, TEST()

### Community 209 - "Technical notes"
Cohesion: 0.12
Nodes (16): Build and CMake, CI, Compilers, cuBLAS and CUDA facts used, Environment, Forward pass vs HF (branch 13, measured Oct 9), HuggingFace BF16 op arithmetic (transformers 5.17.0, torch 2.11.0+cu128, verified by the branch 11 kernels), HuggingFace model facts (transformers 5.17.0, torch 2.11.0+cu128, verified on the RTX 5060) (+8 more)

### Community 210 - "app/layout.tsx"
Cohesion: 0.18
Nodes (11): Web site facts and gotchas (Next.js 16.4.3, motion 14.1.0, verified 2026-10-25 and 26), next-themes, inter, jetbrains, metadata, RootLayout(), Footer(), ScrollToTop() (+3 more)

### Community 211 - "TEST_F"
Cohesion: 0.13
Nodes (8): HttpServer, impl_, Answer, finish_reason, text, generate_answer(), streamed_answer(), TEST_F()

### Community 212 - "chat_api.cpp"
Cohesion: 0.34
Nodes (11): field(), integer(), number(), parse_chat_request(), parse_messages(), parse_stop(), parse_stream(), reject() (+3 more)

### Community 213 - "HttpOptions"
Cohesion: 0.12
Nodes (13): SamplingDefaults, max_tokens, temperature, top_k, top_p, HttpOptions, allowed_hosts, block_size (+5 more)

### Community 214 - "TEST"
Cohesion: 0.20
Nodes (5): host_allowed(), host_name(), lowered(), valid_port(), TEST()

### Community 215 - "MetricsSnapshot"
Cohesion: 0.16
Nodes (12): Percentiles, p50, p99, MetricsSnapshot, blocks_total, counters, output_per_s, time_s (+4 more)

### Community 216 - "Web site plan (Part 2)"
Cohesion: 0.15
Nodes (12): Branches (in order), Constraints, Decisions, Decisions since the plan (Joe, 2026-10-25 and 26), Design tokens, Libraries (from pick-ui-library), Motion rules, Pages (+4 more)

### Community 217 - "Kernels section plan"
Cohesion: 0.17
Nodes (11): Branches (in order), Data, Decisions (Joe, 2026-10-25), Extra pages (Joe, 2026-10-31), How the template is built (for the next pages), Index page `/kernels`, Kernels section plan, Page template (every kernel page) (+3 more)

### Community 218 - "1. Project context and decisions"
Cohesion: 0.17
Nodes (11): 1.1 Who and what, 1.2 Hardware and environment, 1.3 Key technical decisions, 1.4 Critique of the original blueprint (why the plan changed), 1.5 How to build it (AI usage policy), 1.6 Resume impact analysis (honest assessment), 1.7 Learning plan (before and during the build), 1.8 Working preferences for Claude on this project (+3 more)

### Community 219 - "ChatPromptsTest"
Cohesion: 0.18
Nodes (3): ChatPromptsTest, tok, TEST_F()

### Community 220 - "run"
Cohesion: 0.20
Nodes (7): add_bench_command(), format_range(), gpu_name(), graph_buckets(), open_csv(), run(), utc_now()

### Community 221 - "Core concepts you must understand"
Cohesion: 0.18
Nodes (11): 10. Numerics, 1. How a GPU runs code, 2. The memory hierarchy, 3. The roofline model, 4. The Llama forward pass (what you are implementing), 5. Prefill versus decode, 6. The KV cache and why it is paged, 7. Launch overhead and CUDA Graphs (+3 more)

### Community 222 - "Original prompt (verbatim)"
Cohesion: 0.18
Nodes (10): Amendments since (in order), Clarifications Joe answered in the first session, Commits, Delegation, Original prompt (verbatim), Part A: implement fully (I review and test only), Part B: implement as a draft (I will read and explain every line), Part C: leave for me, stub only (+2 more)

### Community 223 - "Working rules"
Cohesion: 0.18
Nodes (10): Code style, Commits, Golden data, Part C stubs, Repo and branches, Reporting, The gate before every commit, The GPU (+2 more)

### Community 224 - "metrics_window.cpp"
Cohesion: 0.35
Nodes (6): MetricsWindow::add_request(), MetricsWindow::add_tokens(), MetricsWindow::output_per_s(), MetricsWindow::tpot_s(), MetricsWindow::ttft_s(), percentiles()

### Community 225 - "TokenEvent"
Cohesion: 0.20
Nodes (7): TokenEvent, completion_tokens, finish, step, text, drain(), text_of()

### Community 226 - "dependencies"
Cohesion: 0.18
Nodes (11): dependencies, clsx, liveline, motion, next, next-themes, @number-flow/react, react (+3 more)

### Community 228 - "devDependencies"
Cohesion: 0.20
Nodes (10): devDependencies, eslint, eslint-config-next, @fontsource/jetbrains-mono, tailwindcss, @tailwindcss/postcss, @types/node, @types/react (+2 more)

### Community 229 - "Branch 73 `server/http` summary (issue #95, engine half)"
Cohesion: 0.22
Nodes (8): Branch 73 `server/http` summary (issue #95, engine half), Commits, Deviations from docs/api.md (all recorded there), Draft PR description, Line-by-line review (Part B), Not verified, Open questions for Joe, Verified end to end (release build, RTX 5060, GPU up)

### Community 230 - "Live"
Cohesion: 0.22
Nodes (9): Live, arrived, delivered, done, events, first_token, generated, stop (+1 more)

### Community 231 - "load_report.py"
Cohesion: 0.44
Nodes (8): aggregate(), curve_plot(), load_run(), main(), meets_slo(), number(), system_of(), table_markdown()

### Community 232 - "scripts"
Cohesion: 0.22
Nodes (9): scripts, build, dev, lint, prebuild, predev, sync-data, test (+1 more)

### Community 233 - "Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95)"
Cohesion: 0.25
Nodes (7): Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95), Commits, Draft PR description, Not verified, needs Joe or the real server, Open questions for Joe, Verified, and how, What it is

### Community 234 - "Testing, correctness, and benchmarking"
Cohesion: 0.25
Nodes (8): Benchmark rules, Debugging method when outputs are wrong, Generating reference data, Metric definitions, Profiling workflow, Test layers, Testing, correctness, and benchmarking, The seven experiments

### Community 235 - "Finished"
Cohesion: 0.21
Nodes (7): Finished, time_s, tpot_s, ttft_s, Tokens, count, time_s

### Community 236 - "Counters"
Cohesion: 0.25
Nodes (8): Counters, blocks_free, finished_total, output_total, preemptions_total, running, steps_total, waiting

### Community 238 - "load_chat_prompts"
Cohesion: 0.29
Nodes (4): Branch 49 `bench/chat-prompts` (worktree ~/code/llm-inference-engine-api), 2 commits on 0d11053, not pushed, Branch 50 `web/scaffold` (worktree ~/code/llm-inference-engine-web), 3 commits on 0d11053, not pushed, Branches 49 and 50 (PR summaries), load_chat_prompts()

### Community 239 - "Part 2: polish, UI/UX, and enhancements"
Cohesion: 0.29
Nodes (7): Part 2: polish, UI/UX, and enhancements, Priority list, Track 1: Serving, Track 2: UI/UX (the web dashboard), Track 3: Runtime features, Track 4: Hard kernels, Track 5: Engineering polish

### Community 240 - "TextStream"
Cohesion: 0.33
Nodes (4): TextStream, decode_, emitted_, ids_

### Community 241 - "@number-flow/react"
Cohesion: 0.48
Nodes (6): @number-flow/react, Num(), Pair(), Tile(), Tiles(), timing

### Community 243 - "row"
Cohesion: 0.33
Nodes (3): RoundTrip, row(), Table

### Community 244 - "Branch 63 `web/polish` (web worktree), from origin/main 1f5d259 (#69 merged), not pushed"
Cohesion: 0.33
Nodes (5): Accessibility pass (CDP, not Lighthouse), Animation review, Branch 63 `web/polish` (web worktree), from origin/main 1f5d259 (#69 merged), not pushed, Break-ui, Open for Joe

### Community 246 - "Step 8: Day-1 smoke tests and troubleshooting"
Cohesion: 0.40
Nodes (5): 8.1 Smoke test checklist, 8.2 Troubleshooting, 8.3 Where to go next, Sources, Step 8: Day-1 smoke tests and troubleshooting

### Community 250 - "Web site"
Cohesion: 0.40
Nodes (4): Deploy on Vercel, Live mode from the deployed site, Run it, Web site

### Community 251 - "KV Block Manager (Branch 15)"
Cohesion: 0.50
Nodes (4): Interface and errors, KV Block Manager (Branch 15), Paged KV cache, Tests

## Knowledge Gaps
- **502 isolated node(s):** `id`, `prompt_tokens`, `output_tokens`, `preemptions`, `arrival_s` (+497 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2429 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **60 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Changelog` connect `Changelog` to `Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)`, `serve_step`, `ModelRunner`, `build-timeline.tsx`, `Profiling`, `scene-figures.tsx`, `Batch`, `ModelWeights`, `PagedRunner`, `add_rmsnorm_vec_kernel`, `SafetensorsHeader`, `[slug]/page.tsx`, `plot.py`, `version`, `swiglu_value`, `BlockManager`, `Scheduler`, `ModelConfig`, `token-rain.tsx`, `write_kv`, `sync-data.mjs`, `Status`, `compare_bf16`, `WorkloadConfig`, `ReferencePrompt`, `parse_safetensors_header`, `rope/story.tsx`, `GpuBackend`, `DType`, `PagedBatch`, `attention-v2/story.tsx`?**
  _High betweenness centrality (0.203) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `useLoop()` (e.g. with `web/kernel-add-rmsnorm` and `web/kernels-template`) actually correct?**
  _`useLoop()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `id`, `prompt_tokens`, `output_tokens` to the rest of the system?**
  _502 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)` be split into smaller, more focused modules?**
  _Cohesion score 0.09246088193456614 - nodes in this community are weakly interconnected._
- **Why does `web/kernels-template` connect `rope/story.tsx` to `[slug]/page.tsx`, `useLoop`, `PagedBatch`, `build-timeline.tsx`, `Changelog`, `app/layout.tsx`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Should `TEST` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Why does `Part 1 - Engine (M1 Foundations)` connect `Part 1 - Engine (M1 Foundations)` to `Buffer (Task 1)`, `ModelConfig`, `Scheduler (Branch 19)`, `write_kv Implementation (kernel/write-kv-impl)`, `Status`, `Docs (Branch 8)`, `Paged Decode Attention (Branch 17)`, `.update`, `Repo Housekeeping (Branch 9)`, `GpuBackend`, `Naive Oracle Kernels (Branch 11)`, `Forward Pass (Branch 13)`, `paged_attention_decode Implementation (kernel/paged-attention-impl)`, `Batched Sampling (Branch 18)`, `Tokenizer (Branch 10)`, `Weight Upload (Branch 6)`, `Sampling (Branch 14)`, `Block Manager Implementation (kv/block-manager-impl)`, `Naive Attention (Branch 12)`, `Bench Harness (Branch 21)`, `Tensor and Buffers (Branch 2)`, `Paged KV Cache (Branch 16)`, `Checksums (Branch 7)`, `cuBLAS GEMM (Branch 3)`, `KV Block Manager (Branch 15)`, `Build Scaffolding (Branch 1)`, `Safetensors Loader Stub (Branch 5)`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._