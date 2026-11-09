# Graph Report - llm-inference-engine  (2026-11-09)

## Corpus Check
- 599 files · ~1,096,677 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 309 file(s) not represented in the graph (top: .csv 282, (none) 26, .css 1)

## Summary
- 6899 nodes · 15775 edges · 323 communities (250 shown, 73 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1173 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `673d184a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)
- chapter.ts
- tinyllama_advanced.py
- algorithm
- ModelRunner
- results.ts
- Review notes
- Safe Tensor Reader (Task 2)
- TEST
- vector
- .data
- TEST
- build-timeline.tsx
- TEST
- WeightLayout
- TEST
- TEST
- string
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
- reading-view.tsx
- Options
- TEST
- front-matter.ts
- visualizer.tsx
- Scheduler
- ModelConfig
- TEST
- write_kv
- RequestMetrics
- Summary
- TEST_P
- tokenizer.cpp
- sync-data.mjs
- PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)
- StepBroadcaster
- story-view.tsx
- mapped_file.cpp
- react
- TextStream
- block_manager.cpp
- TEST
- run
- TEST_F
- WeightUploadTest
- Status
- compare_bf16
- Options
- gpu-floorplan.tsx
- foundations/story.tsx
- cuda-graphs/story.tsx
- make_workload
- WorkloadConfig
- Tokenizer
- README.md
- kernels/types.ts
- ReferencePrompt
- SchedulerConfig
- PinnedBuffer
- Label
- Part 1 - Engine (M1 Foundations)
- stream.tsx
- TEST
- BookScene
- reviews/index.ts
- chunked-prefill/sandbox.tsx
- foundations/sandbox.tsx
- TEST_F
- Changelog
- bandwith_test.cu
- book-scene.ts
- rope/story.tsx
- paged-kv/story.tsx
- baseline_llamacpp.py
- TEST_F
- GpuBackend
- onchip_bench.cu
- TEST_P
- TEST
- Naive Oracle Kernels (Branch 11)
- Forward Pass (Branch 13)
- paged_attention_decode Implementation (kernel/paged-attention-impl)
- sampling-model.ts
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
- split-attention/story.tsx
- PagedBatch
- line-chart.tsx
- chat_prompt
- results/story.tsx
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
- math
- ServingLoop
- TEST
- Build Scaffolding (Branch 1)
- attention-v2/story.tsx
- Safetensors Loader Stub (Branch 5)
- serving_loop.cpp
- profile_report.py
- serve_step
- Shape
- SafetensorsFile
- data.ts
- TEST_F
- sampling/story.tsx
- DashModel
- live.ts
- add-rmsnorm/story.tsx
- TEST
- TEST
- TEST
- baseline_vllm.py
- TEST
- attention_kernel
- LLM Inference Engine Setup Guide
- Profiling
- measured
- scene-figures.tsx
- tensor.h
- Shape
- bandwith_test.md
- dram_throughput.md
- format.sh
- paged_attention_harness.h
- Blocks
- TEST_F
- trace.ts
- add_rmsnorm.cu
- TEST
- sampling/sandbox.tsx
- StopMatcher
- argparse
- version
- test_swiglu_fast.cpp
- RunGrouping
- Completion
- usePrefersReducedMotion
- useLoop
- HttpServer::Impl
- decode_step_b16.md
- Plot
- kv-pool.tsx
- sampling/about.tsx
- Results
- Scheduler (Branch 19)
- Attribution
- write_kv Implementation (kernel/write-kv-impl)
- inkSegments
- BlockManager
- TEST
- reproduce.sh
- LlamaBench
- test_baseline_vllm.py
- decode_step.md
- decode_step_graphs.md
- rope_kernel
- paged-kv/figures.tsx
- TEST
- TEST
- Docs (Branch 8)
- gemm/sandbox.tsx
- Paged Decode Attention (Branch 17)
- createCard
- setup_llamacpp.sh
- profile.sh
- bench
- Options
- TEST
- Repo Housekeeping (Branch 9)
- Part 1 (MVP): component specs
- Pending
- Feed
- cellNumber
- compilerOptions
- three
- MetricsWindow
- app/layout.tsx
- gloss.tsx
- run
- chat_api.cpp
- HttpOptions
- cuda-graphs-model.ts
- MetricsSnapshot
- Web site plan (Part 2)
- Kernels section plan
- 1. Project context and decisions
- load_chat_prompts
- TEST
- Core concepts you must understand
- parse_safetensors_header
- Working rules
- metrics_window.cpp
- SmoothRange
- dependencies
- LoadReport
- exercise-card.tsx
- Branch 73 `server/http` summary (issue #95, engine half)
- .run_staged
- load_report.py
- paged-kv-model.ts
- Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95)
- LLM Inference Engine Build Plan
- BlockManager
- rope/sandbox.tsx
- TEST
- Kit
- Part 2: polish, UI/UX, and enhancements
- rope-model.ts
- attention/sandbox.tsx
- LlamaBatchedBench
- row
- add-rmsnorm-model.ts
- washPlane
- Step 8: Day-1 smoke tests and troubleshooting
- Metrics
- Workload
- ProfileTest
- Web site
- KV Block Manager (Branch 15)
- utility
- Correctness
- load_sweep.sh
- setup_vllm.sh
- Driver
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
- add-rmsnorm-about-model.ts
- add-rmsnorm/about.tsx
- TEST
- expect_invariants
- scheduler.cpp
- paged-attention-about-model.ts
- swiglu-model.ts
- Pool
- ChunkedStep
- gemm-model.ts
- swiglu-about-model.ts
- model_runner.cpp
- prefill_ttft.py
- attention-v2-model.ts
- split-attention-about-model.ts
- sample.cu
- View
- foundations-about-model.ts
- Branch 84: bench/chunked-prefill (issue #101, Part A)
- Branch 98: web/sandbox-perf
- Branch 90: tools/onchip-speeds
- FakeRunner
- Table
- scoreboard.tsx
- Step 1: Windows host prep (you, ~45 min)
- Exercise: sum reduction of 16M floats
- Step 5: Partner's Mac setup and remote GPU access (~1.5 hours)
- Step 7: CI and team conventions (together, ~45 min)
- rope_kernel
- Part 1 (MVP): schedule and milestones
- Resume bullets, README, and interview prep
- On-chip Memory Speeds (Branch 90)
- test_version.cpp
- paged_attention_split_merge_kernel
- Sampler::operator()
- TEST
- TEST
- add_kernel
- embedding_kernel
- paged_attention_decode_v2_kernel
- rmsnorm_kernel
- swiglu_kernel
- branch86_summary.md
- branch88_summary.md
- branch91_summary.md
- branch92_summary.md
- branch93_summary.md
- branch94_summary.md
- branch95_summary.md
- branch96_summary.md

## God Nodes (most connected - your core abstractions)
1. `useLoop()` - 154 edges
2. `Label()` - 153 edges
3. `Part 1 - Engine (M1 Foundations)` - 136 edges
4. `Frame()` - 133 edges
5. `Changelog` - 133 edges
6. `washPlane()` - 103 edges
7. `createCard()` - 79 edges
8. `Tensor` - 70 edges
9. `react` - 70 edges
10. `measured()` - 70 edges

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

## Communities (323 total, 73 thin omitted)

### Community 0 - "Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)"
Cohesion: 0.10
Nodes (24): tools/checksum, Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge), TensorChecksum, abs_sum, dtype, name, shape, sum (+16 more)

### Community 1 - "chapter.ts"
Cohesion: 0.03
Nodes (93): web/book-about, web/book-rope, BookChapter(), Beat, Frame, Stop, ChapterDef, chapterStops() (+85 more)

### Community 2 - "tinyllama_advanced.py"
Cohesion: 0.11
Nodes (22): capture(), keep(), chat_ids(), dump_prompt(), main(), main(), apply_rope(), forward() (+14 more)

### Community 4 - "ModelRunner"
Cohesion: 0.05
Nodes (29): runtime/wire-fast-kernels, tools/bench, Branch 20 summary: model/paged-runner (Part B), Branch 21: tools/bench, Decisions (confirmed by Joe on Oct 11), Measured (RTX 5060, release, Oct 11), Usage, Worth a look (+21 more)

### Community 5 - "results.ts"
Cohesion: 0.06
Nodes (38): batchSizes, benchSeries(), benchSystems, benchTable, checkedMedian(), contextSeries(), contextTpot, docCell() (+30 more)

### Community 6 - "Review notes"
Cohesion: 0.12
Nodes (14): Check against the dashboard (Branch 75's list), Branch 13: model/forward (Part B wiring, greedy and generate are line-by-line review items; TextStream and the CLI are Part A), Branch 14: sampling/temperature (Part B kernel and generate are line-by-line review items; the CLI flags are Part A), Branch 17: kernel/paged-decode-attention (Part C stub; nothing for line-by-line review), Branch 19: sched/scheduler (Part A `Request` plus a Part C stub; nothing for line-by-line review), Branch 21: tools/bench (Part A; nothing for line-by-line review, two spots worth a look), Branch 2: core/tensor-and-buffers, Branch 3: kernel/cublas-gemm (Part B, line-by-line review item) (+6 more)

### Community 7 - "Safe Tensor Reader (Task 2)"
Cohesion: 0.07
Nodes (28): Atomics and shared memory, Bandwith Test (Task 3/CUDA Task 1), Buffer (Task 1), Build, Bytes and types, Core C++, Core Ideas, Core Ideas (+20 more)

### Community 9 - "TEST"
Cohesion: 0.06
Nodes (8): as_bytes(), safetensors_file(), read(), entry(), expect_rejected(), expect_rejected_header(), parse(), TEST()

### Community 10 - "vector"
Cohesion: 0.11
Nodes (5): OwnedStream, stream, TextStream::push(), model_next(), solo()

### Community 11 - ".data"
Cohesion: 0.06
Nodes (19): DeviceBuffer, ptr_, DeviceBuffer::DeviceBuffer(), TEST(), TEST(), download(), GemmShapeTest, GemmTest (+11 more)

### Community 12 - "TEST"
Cohesion: 0.05
Nodes (29): Branch 18: sampling/batched (Part B kernel, Sampler and generate are line-by-line review items; the CLI flags are Part A), SamplingParams, seed, temperature, top_k, top_p, batch, validate() (+21 more)

### Community 13 - "build-timeline.tsx"
Cohesion: 0.05
Nodes (59): web/overview, web/polish, Accessibility pass (CDP, not Lighthouse), Animation review, Branch 63 `web/polish` (web worktree), from origin/main 1f5d259 (#69 merged), not pushed, Break-ui, Open for Joe, cards (+51 more)

### Community 14 - "TEST"
Cohesion: 0.11
Nodes (11): Branch 16: kv/paged-cache (Part A plus a Part C stub; nothing for line-by-line review), Case, block_size, head_dim, heads, kv_heads, num_blocks, check() (+3 more)

### Community 15 - "WeightLayout"
Cohesion: 0.07
Nodes (29): CopyOp, bytes, dst, name, shape, src, LayerLayout, ln1 (+21 more)

### Community 17 - "TEST"
Cohesion: 0.09
Nodes (19): data, attention_scale(), AttentionLayoutTest, Cache, l_, rows_, cpu_attention(), Expected (+11 more)

### Community 18 - "string"
Cohesion: 0.09
Nodes (5): span, word_decode(), bf16_bytes(), bytes_of(), always_11()

### Community 19 - "pathlib"
Cohesion: 0.06
Nodes (17): engine_command(), main(), measure(), prompts_with_a_long_one(), table_markdown(), row(), Scoreboard, Parse (+9 more)

### Community 20 - "MappedFile"
Cohesion: 0.10
Nodes (9): MappedFile, data_, size_, fail(), MappedFile::MappedFile(), TEST(), TempFile, path_ (+1 more)

### Community 21 - "Request"
Cohesion: 0.10
Nodes (15): Request, arrived, finished, first_token, id, max_new_tokens, output, params (+7 more)

### Community 22 - "CudaStream"
Cohesion: 0.11
Nodes (6): generate(), GenerateTest, stream, TEST_F(), CudaStream, stream_

### Community 23 - "from_bf16"
Cohesion: 0.05
Nodes (25): cpu_add(), AddRmsnormShapeTest, cpu_add(), cpu_rmsnorm(), Result, norm, sum, run() (+17 more)

### Community 24 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, hidden, shift, tokens

### Community 26 - "ModelWeights"
Cohesion: 0.06
Nodes (32): loader/weight-upload, Branch 6: loader/weight-upload (the upload is a line-by-line review item), HTTP Server (Branch 73), LayerWeights, ln1, ln2, wdown, wgu (+24 more)

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
Cohesion: 0.13
Nodes (17): loader/safetensors-stub, dtype_size(), SafetensorsHeader, data_offset, tensors, TensorEntry, bytes, dtype (+9 more)

### Community 32 - "Tensor"
Cohesion: 0.08
Nodes (15): Tensor, dtype, kMaxDims, ndim, shape, add(), same_shape(), rope() (+7 more)

### Community 33 - "reading-view.tsx"
Cohesion: 0.05
Nodes (60): Branch 87: web/prefill-decode-prologue (issue #136), next, config, BookChapterPage(), dynamicParams, generateMetadata(), Exercises(), groups (+52 more)

### Community 34 - "Options"
Cohesion: 0.09
Nodes (21): open_csv(), Options, graphs, max_batch, model, num_requests, out, output_len (+13 more)

### Community 35 - "TEST"
Cohesion: 0.09
Nodes (7): blocks_for(), expect_consistent(), Snapshot, free, lengths, tables, TEST()

### Community 36 - "front-matter.ts"
Cohesion: 0.05
Nodes (84): web/book-cover, Contents(), metadata, Kernels(), metadata, metadata, Start(), OverlayRect (+76 more)

### Community 37 - "visualizer.tsx"
Cohesion: 0.08
Nodes (33): PagedKvOverlay(), Cell, Mode, Seq, Shape, BlockDetails(), Colors, corners() (+25 more)

### Community 38 - "Scheduler"
Cohesion: 0.11
Nodes (13): sched/scheduler, Decisions in branch 19 (Claude's, open to Joe's change), Scheduler, block_manager_, config_, finished_list_, pending_sampled_seqs_, pending_step_ (+5 more)

### Community 39 - "ModelConfig"
Cohesion: 0.06
Nodes (32): model/config, Branches, How Joe verifies a branch, M1 plan, What each branch contains, Decisions in branch 16 (Claude's, open to Joe's change), Config semantics, Model Config (Branch 4) (+24 more)

### Community 41 - "write_kv"
Cohesion: 0.07
Nodes (33): docs/write-kv-profile, kernel/write-kv-impl, sampling/batched, sampling/temperature, tools/profiling, tools/sample-bench, C3. Kernels (weeks 2 to 6), Branches (order confirmed by Joe on Oct 10) (+25 more)

### Community 42 - "RequestMetrics"
Cohesion: 0.12
Nodes (13): RequestMetrics, arrival_s, id, latency_s, output_tokens, preemptions, prompt_tokens, tpot_s (+5 more)

### Community 43 - "Summary"
Cohesion: 0.13
Nodes (15): Stat, mean, p50, p90, p99, Summary, latency_s, output_throughput (+7 more)

### Community 44 - "TEST_P"
Cohesion: 0.19
Nodes (9): cpu_rope(), gpu_rope(), Layout, head_dim, heads, kv_heads, RopeLayoutTest, spread_positions() (+1 more)

### Community 45 - "tokenizer.cpp"
Cohesion: 0.27
Nodes (5): fail(), Tokenizer::append_text(), Tokenizer::decode(), Tokenizer::encode(), Tokenizer::Tokenizer()

### Community 46 - "sync-data.mjs"
Cohesion: 0.04
Nodes (55): web/kernel-cuda-graphs, Branch 59 `web/kernel-cuda-graphs` (web worktree), from origin/main 03fefac (#65 merged), not pushed, path, main(), out, parseSpecs(), specSources, splitLabel() (+47 more)

### Community 47 - "PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)"
Cohesion: 0.10
Nodes (18): Calling cuBLAS with row-major weights, Model and system architecture, Model config ([config.json](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json)), Repository layout, System architecture, Tokenizer, Weight tensors (safetensors names and shapes), Attention (+10 more)

### Community 48 - "StepBroadcaster"
Cohesion: 0.05
Nodes (14): StepBroadcaster, capacity_, closed_, mutex_, subscribers_, Channel, capacity_, items_ (+6 more)

### Community 49 - "story-view.tsx"
Cohesion: 0.05
Nodes (75): web/kernels-template, Branch 74 `web/results` (issue #92), summary, Commits, Draft PR description, Not verified, Open questions for Joe, Verified, What was built (+67 more)

### Community 50 - "mapped_file.cpp"
Cohesion: 0.12
Nodes (6): main(), print_version(), Fd, fd, SafetensorReader, main()

### Community 51 - "react"
Cohesion: 0.10
Nodes (34): clsx, react, DashboardPage(), metadata, recordingLabel(), Code(), ConnectBar(), ConnectHelp() (+26 more)

### Community 52 - "TextStream"
Cohesion: 0.12
Nodes (7): TextStream, decode_, emitted_, ids_, TEST_F(), TextStreamTest, tok

### Community 53 - "block_manager.cpp"
Cohesion: 0.17
Nodes (6): BlockManager::allocate(), BlockManager::append_slot(), BlockManager::can_append(), BlockManager::free(), BlockManager::length(), BlockManager::slot()

### Community 54 - "TEST"
Cohesion: 0.12
Nodes (3): parse_length_range(), parse_positive(), TEST()

### Community 55 - "run"
Cohesion: 0.16
Nodes (11): add_bench_command(), format_range(), gpu_name(), graph_buckets(), run(), utc_now(), csv_escape(), csv_number() (+3 more)

### Community 56 - "TEST_F"
Cohesion: 0.18
Nodes (3): BenchTest, stream, TEST_F()

### Community 57 - "WeightUploadTest"
Cohesion: 0.17
Nodes (5): TEST_P(), WeightUploadTest, file_, layout_, stream_

### Community 58 - "Status"
Cohesion: 0.08
Nodes (24): kernel/paged-attention-impl, kernel/paged-decode-attention, M1 verification (Joe's run, Oct 9, in the Claude worktree), Milestones, Next steps for Claude, Part C stubs that exist, Snapshot 2026-10-26 (after midnight): CUDA Graphs page, Snapshot 2026-10-26 (end of the web overview + first kernel story session) (+16 more)

### Community 59 - "compare_bf16"
Cohesion: 0.06
Nodes (18): kernel/add-rmsnorm-impl, kernel/naive-oracles, Branch 11: kernel/naive-oracles (Part B, every line is a review item), rmsnorm(), TEST(), cpu_rmsnorm(), gpu_rmsnorm(), RmsnormShapeTest (+10 more)

### Community 60 - "Options"
Cohesion: 0.13
Nodes (11): add_generate_command(), Options, max_new_tokens, model, prompt, seed, system, temperature (+3 more)

### Community 61 - "gpu-floorplan.tsx"
Cohesion: 0.08
Nodes (41): A, B, bandwidth, blockTable, Box, bx(), C, checksum (+33 more)

### Community 62 - "foundations/story.tsx"
Cohesion: 0.04
Nodes (74): blockBox, Box, ContrastFigure(), FusionFigure(), FusionNumbers, gridBox(), gridCell, HierarchyFigure() (+66 more)

### Community 63 - "cuda-graphs/story.tsx"
Cohesion: 0.10
Nodes (26): CudaGraphsOverlay(), CaptureFigure(), EagerTimelineFigure(), fmt(), GapFigure(), GraphGridFigure(), graphLayer, LaneLayer() (+18 more)

### Community 64 - "make_workload"
Cohesion: 0.21
Nodes (7): LengthRange, max, min, check(), make_workload(), SplitMix64, state_

### Community 65 - "WorkloadConfig"
Cohesion: 0.11
Nodes (16): bench/chat-prompts, Workload, arrivals, requests, WorkloadConfig, bos_id, eos_id, num_requests (+8 more)

### Community 66 - "Tokenizer"
Cohesion: 0.12
Nodes (9): Branch 10: tokenizer/sentencepiece (Part A), SentencePieceProcessor, Tokenizer, sentinel_, sp_, unk_id_, ServerTest, tok (+1 more)

### Community 67 - "README.md"
Cohesion: 0.15
Nodes (12): Architecture, Beyond the MVP, Building, Design decisions, Goals, Hardware, Ownership, Repository layout (+4 more)

### Community 68 - "kernels/types.ts"
Cohesion: 0.05
Nodes (43): KernelCards(), choices, maxShare, shareIn(), Metric, DeadEnd, DeadEndKind, kindLabel (+35 more)

### Community 69 - "ReferencePrompt"
Cohesion: 0.11
Nodes (18): tokenizer/sentencepiece, Reference format (`tests/golden/reference/`, generated, gitignored), golden_checksums_path(), golden_dir(), model_dir(), load_reference_manifest(), reference_dir(), reference_skip_reason() (+10 more)

### Community 70 - "SchedulerConfig"
Cohesion: 0.15
Nodes (12): Done but not merged, SchedulerConfig, block_size, eos_id, max_batch, max_context, max_prefill_tokens, num_blocks (+4 more)

### Community 71 - "PinnedBuffer"
Cohesion: 0.22
Nodes (4): PinnedBuffer, ptr_, PinnedBuffer::PinnedBuffer(), TEST()

### Community 72 - "Label"
Cohesion: 0.06
Nodes (65): FoundationsOverlay(), at(), ChunkedFigure(), Lane(), stall, ticks, AllPrefillFigure(), BudgetBarFigure() (+57 more)

### Community 73 - "Part 1 - Engine (M1 Foundations)"
Cohesion: 0.02
Nodes (108): 3D book: add and RMSNorm (Branch 108), 3D book: attention v2 (Branch 106), 3D book behind a tab (Branch 113), 3D book (Branch 100), 3D book: chapter intros (Branch 117), 3D book: chunked prefill (Branch 112), 3D book: cover and contents (Branch 101), 3D book: CUDA Graphs (Branch 111) (+100 more)

### Community 74 - "stream.tsx"
Cohesion: 0.07
Nodes (56): aim(), at(), Cell, closest, Colors, corners(), cost(), drawList (+48 more)

### Community 75 - "TEST"
Cohesion: 0.20
Nodes (3): cpu_greedy(), run_greedy(), TEST()

### Community 76 - "BookScene"
Cohesion: 0.11
Nodes (9): BookScene, clamp01(), easeInOut(), edgeTexture(), pageGeometry(), pageHeight(), rad(), TextSide (+1 more)

### Community 77 - "reviews/index.ts"
Cohesion: 0.06
Nodes (38): review, after, before, dram, review, v2Bytes, review, eager (+30 more)

### Community 78 - "chunked-prefill/sandbox.tsx"
Cohesion: 0.06
Nodes (44): chunks(), run(), aloneLarge, aloneSmall, budgets, cellClass(), cellName, guessStall (+36 more)

### Community 79 - "foundations/sandbox.tsx"
Cohesion: 0.07
Nodes (44): Root causes, with evidence, ceilings, context, results, shape, analyse(), Ceilings, Cost (+36 more)

### Community 80 - "TEST_F"
Cohesion: 0.14
Nodes (3): TEST_F(), TokenizerTest, tok

### Community 81 - "Changelog"
Cohesion: 0.03
Nodes (71): bench/chunked-prefill, bench/prefill-ttft, bench/qps-sweep, bench/vllm-baseline, build/ci-timeouts, build/repo-housekeeping, Changelog, docs/attention-v2-readme (+63 more)

### Community 82 - "bandwith_test.cu"
Cohesion: 0.36
Nodes (6): launch(), main(), timeKernel(), upload(), vectorAdd(), verify()

### Community 83 - "book-scene.ts"
Cohesion: 0.07
Nodes (40): web/glossary, BLOCK, defaultCoverFrame, easeOutBack(), OverlayKind, PAGE_D, PAGE_W, PageLink (+32 more)

### Community 84 - "rope/story.tsx"
Cohesion: 0.10
Nodes (28): RopeOverlay(), BitsFigure(), dials, heads, LayoutFigure(), ParallelFigure(), RotateFigure(), SerialFigure() (+20 more)

### Community 85 - "paged-kv/story.tsx"
Cohesion: 0.08
Nodes (33): beats, chapter, demo, instructions, pagedKvChapter, workbench, first, fullContext (+25 more)

### Community 86 - "baseline_llamacpp.py"
Cohesion: 0.10
Nodes (30): greedy_batch(), main(), run(), batched_command(), batched_main(), bench_main(), gpu_name(), main() (+22 more)

### Community 87 - "TEST_F"
Cohesion: 0.06
Nodes (20): DecodeGraphs, buckets_, graphs_, runner_, Arrival, at_iteration, id, max_new_tokens (+12 more)

### Community 88 - "GpuBackend"
Cohesion: 0.09
Nodes (15): StepResult, finished, trace, GpuBackend, cache_, config_, graphs_, next_ (+7 more)

### Community 89 - "onchip_bench.cu"
Cohesion: 0.14
Nodes (28): add(), bandwidth_rows(), chase_global(), chase_shared(), check(), cycle(), Device, out (+20 more)

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

### Community 95 - "sampling-model.ts"
Cohesion: 0.12
Nodes (36): approach(), column(), colX(), createSamplingAboutModel(), fmt(), nameOf(), pct(), Pen (+28 more)

### Community 96 - "TEST_F"
Cohesion: 0.07
Nodes (12): byte_encode(), chat_body(), HttpServerTest, backend, engine, http, listener, loop (+4 more)

### Community 97 - "ChatRequest"
Cohesion: 0.11
Nodes (11): ApiError, ChatRequest, max_tokens, messages, params, seed, stop, stream (+3 more)

### Community 99 - "DType"
Cohesion: 0.07
Nodes (21): core/tensor-and-buffers, Cadence, Git workflow, Interfaces to agree on in week 0, Ownership, Team split and workflow, CUstream_st, DType (+13 more)

### Community 100 - "StepTrace"
Cohesion: 0.07
Nodes (17): Branch 48 `docs/api-contract` (PR summary), StepTrace, end, finished, free_blocks, prefill, running, seqs (+9 more)

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
Cohesion: 0.16
Nodes (15): Cloud GPUs (only for final benchmarks), Hardware and toolchain, Learning path (both of you, weeks 0 to 2), Operating system setup, Risks, Risks, mitigations, and learning resources, Sources, Toolchain (pin these versions in the README) (+7 more)

### Community 106 - "split-attention/story.tsx"
Cohesion: 0.06
Nodes (48): beats, captions, chapter, instructions, splitAttentionChapter, SplitOverlay(), splitAbout, splitAboutCaption (+40 more)

### Community 107 - "PagedBatch"
Cohesion: 0.06
Nodes (16): PagedBatch, block_tables, context_lens, logit_rows, max_blocks, num_seqs, positions, prefill (+8 more)

### Community 108 - "line-chart.tsx"
Cohesion: 0.12
Nodes (25): Axis, LineChart(), pad, pathOf(), Props, Readout(), Row, scale() (+17 more)

### Community 109 - "chat_prompt"
Cohesion: 0.13
Nodes (7): Branches, Decisions Joe confirmed on Oct 9 ("yes to all"), M2 plan: correct naive engine (due Oct 18), Open questions for later in M2, Results (branch 13, Oct 9), chat_prompt(), TEST()

### Community 110 - "results/story.tsx"
Cohesion: 0.06
Nodes (34): batchAxis, benchStats(), chunked, contextAxis, contextY, flopsAxis, goodputAxis, intensityAxis (+26 more)

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

### Community 121 - "math"
Cohesion: 0.17
Nodes (24): aggregate(), bandwidth_plot(), line_plot(), load_summaries(), main(), _place(), series(), spread_labels() (+16 more)

### Community 122 - "ServingLoop"
Cohesion: 0.06
Nodes (18): Backend, ServingLoop, cancels_, counters_, decode_, eos_id_, inbox_mutex_, inbox_ready_ (+10 more)

### Community 123 - "TEST"
Cohesion: 0.11
Nodes (9): check(), consecutive(), Layout, head_dim, heads, kv_heads, random_positions(), TEST() (+1 more)

### Community 124 - "Build Scaffolding (Branch 1)"
Cohesion: 0.50
Nodes (4): Build Scaffolding (Branch 1), CI and tooling, CMake and dependencies, Python

### Community 125 - "attention-v2/story.tsx"
Cohesion: 0.07
Nodes (48): attentionV2Chapter, AttentionV2Overlay(), beats, chapter, instructions, attentionV2About, attentionV2AboutCaption, AttentionV2AboutFigure() (+40 more)

### Community 126 - "Safetensors Loader Stub (Branch 5)"
Cohesion: 0.50
Nodes (4): Mapping files, Safetensors Loader Stub (Branch 5), Safetensors parsing (your Part C), Testing

### Community 127 - "serving_loop.cpp"
Cohesion: 0.09
Nodes (20): Live, arrived, delivered, done, events, first_token, generated, stop (+12 more)

### Community 128 - "profile_report.py"
Cohesion: 0.16
Nodes (18): Branch 15: kv/block-manager (Part C stub; nothing for line-by-line review), attribute(), bandwidth_section(), busy_and_span(), export_sqlite(), fmt(), kernel_table(), load_kernels() (+10 more)

### Community 129 - "serve_step"
Cohesion: 0.08
Nodes (28): bench/batched, docs/api-contract, runtime/cuda-graphs-impl, sched/chunked-prefill-wiring, Draft PR description, What changed, Open items and notes, NvtxRange (+20 more)

### Community 130 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, b, in, out

### Community 131 - "SafetensorsFile"
Cohesion: 0.06
Nodes (11): Headline finding: outputs match up to BF16 ties, not token for token, TEST(), TEST(), to_host(), ServeTest, prompts, stream, TEST_F() (+3 more)

### Community 132 - "data.ts"
Cohesion: 0.07
Nodes (16): batches, gapFiles, KvDemoRequest, KvRequest, median(), onchip, OnchipRow, onchipRows (+8 more)

### Community 133 - "TEST_F"
Cohesion: 0.12
Nodes (5): PagedRunnerTest, stream, prompt_of(), TEST_F(), to_host()

### Community 134 - "sampling/story.tsx"
Cohesion: 0.15
Nodes (24): ArgmaxFigure(), Bars(), CostFigure(), Fp64Figure(), GumbelFigure(), Mode, PhiloxFigure(), searchSteps() (+16 more)

### Community 136 - "live.ts"
Cohesion: 0.13
Nodes (13): Blocked, getJson(), HttpError, isLoopback(), isPrivateHost(), LiveConnection, LiveStatus, Metrics (+5 more)

### Community 137 - "add-rmsnorm/story.tsx"
Cohesion: 0.09
Nodes (31): AddRmsnormOverlay(), FirstFusedFigure(), fusedKb64, JobFigure(), lanesStart, MeasuredBars(), naiveKb64, Packet() (+23 more)

### Community 140 - "TEST"
Cohesion: 0.06
Nodes (14): CorsPolicy, origins_, CorsPolicy::allows(), CorsPolicy::CorsPolicy(), CorsPolicy::preflight_headers(), CorsPolicy::response_headers(), validate_origin(), host_allowed() (+6 more)

### Community 141 - "baseline_vllm.py"
Cohesion: 0.11
Nodes (24): completion_payload(), drive(), fetch(), gpu_name(), main(), parse_preemptions(), parse_rates(), parse_server_log() (+16 more)

### Community 144 - "LLM Inference Engine Setup Guide"
Cohesion: 0.13
Nodes (14): 2.1 Base packages, 2.2 CUDA toolkit (WSL-Ubuntu repo), 2.3 Environment variables, 2.4 Verify the GPU, 2.5 GitHub access from WSL, 4.1 Create the virtual environment, 4.2 Download TinyLlama, 4.3 Verify PyTorch and the model (+6 more)

### Community 145 - "Profiling"
Cohesion: 0.06
Nodes (35): kernel/paged-attention-split-impl, kernel/paged-attention-split-stub, kernel/paged-attention-v2-impl, runtime/wire-attention-v2, runtime/wire-attention-v2, Attention shared memory (Branch 67), Decode CUDA Graphs (Branch 34), Fast kernels in the forward pass (Branch 36) (+27 more)

### Community 146 - "measured"
Cohesion: 0.06
Nodes (53): Branch 85: web/chunked-prefill-story, beats, chapter, instructions, swigluChapter, SwigluOverlay(), Author(), InBook (+45 more)

### Community 147 - "scene-figures.tsx"
Cohesion: 0.11
Nodes (25): web/kernel-gemm, Branch 58 `web/kernel-gemm` (web worktree), from origin/main 7e8df6b (#64 merged), not pushed, metric(), batch1, batch16, decode, GenerateFigure(), logits (+17 more)

### Community 148 - "tensor.h"
Cohesion: 0.07
Nodes (11): dtype_name(), main(), scale_kernel(), write_kv_kernel(), dtype_from_name(), sumNaiveAtomics(), sumSharedMemory(), sumWarpShuffle() (+3 more)

### Community 149 - "Shape"
Cohesion: 0.67
Nodes (3): Shape, hidden, tokens

### Community 154 - "paged_attention_harness.h"
Cohesion: 0.06
Nodes (35): kernel/paged-attention-v2-stub, check(), PagedLayoutTest, run_paged(), TEST(), TEST_P(), PagedLayoutV2Test, TEST_P() (+27 more)

### Community 156 - "TEST_F"
Cohesion: 0.18
Nodes (3): BatchedBenchTest, stream, TEST_F()

### Community 157 - "trace.ts"
Cohesion: 0.14
Nodes (19): Num(), Pair(), Tile(), Tiles(), timing, emptyTiles, percentile(), replayTiles() (+11 more)

### Community 158 - "add_rmsnorm.cu"
Cohesion: 0.36
Nodes (6): kernel/add-rmsnorm-tune, add_rmsnorm_kernel(), add_rmsnorm_vec_kernel(), blockAllReduceSum(), blockReduceSum(), warpReduceSum()

### Community 159 - "TEST"
Cohesion: 0.11
Nodes (7): TEST(), GuardedBf16, begin_, buffer_, kGuard, kSentinel, n_

### Community 160 - "sampling/sandbox.tsx"
Cohesion: 0.10
Nodes (32): dump, groups, off, oddsOf(), buildGroups(), draw(), Dump, entropyBits() (+24 more)

### Community 161 - "StopMatcher"
Cohesion: 0.10
Nodes (9): Result, stopped, text, StopMatcher, held_, stops_, StopMatcher::push(), run() (+1 more)

### Community 162 - "argparse"
Cohesion: 0.17
Nodes (14): docs/results, checksums(), main(), best(), environment_block(), main(), median_of(), parse_correctness() (+6 more)

### Community 163 - "version"
Cohesion: 0.11
Nodes (15): build/m1-scaffolding, Baselines and plots, HuggingFace transformers, llama.cpp, Plots, Reading the comparison fairly, The workload, vLLM (+7 more)

### Community 164 - "test_swiglu_fast.cpp"
Cohesion: 0.12
Nodes (16): kernel/swiglu-small-batch, kernel/swiglu-vec-impl, sm_count_of_current_device(), swiglu(), swiglu_kernel(), swiglu_value(), swiglu_vec_kernel(), check() (+8 more)

### Community 166 - "Completion"
Cohesion: 0.12
Nodes (21): Completion, created, model, seq, FinishReason, Length, Stop, chunk() (+13 more)

### Community 167 - "usePrefersReducedMotion"
Cohesion: 0.07
Nodes (34): Branch 52 `web/overview` (worktree ~/code/llm-inference-engine-web), 4 commits STACKED on web/prototypes (1faf8ab), not pushed, Before and after (median of 3 runs, local static export, headless Chrome with GPU, 1280x800, cold HTTP cache), Branch 99: web/nav-speed (page navigation speed), Commits, Conclusion, Draft PR description, How it was measured (scripts in ~/.cache/llmie-nav/), Not verified (+26 more)

### Community 168 - "useLoop"
Cohesion: 0.03
Nodes (86): beats, chapter, instructions, pagedAttentionChapter, PagedAttentionOverlay(), workbench, pagedAttentionAbout, pagedAttentionAboutCaption (+78 more)

### Community 169 - "HttpServer::Impl"
Cohesion: 0.19
Nodes (11): error_json(), HttpServer::HttpServer(), HttpServer::Impl, cors, encode, http, options, random_seed() (+3 more)

### Community 170 - "decode_step_b16.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 172 - "kv-pool.tsx"
Cohesion: 0.10
Nodes (32): Colors, readColors(), slotColor(), SLOTS, BlockDetails(), Cell, fmt(), Geo (+24 more)

### Community 173 - "sampling/about.tsx"
Cohesion: 0.07
Nodes (30): aboutNumbers, bars, fmt(), foundationsAbout, foundationsAboutCaption, FoundationsAboutFigure(), FoundationsAboutStory(), graphs (+22 more)

### Community 174 - "Results"
Cohesion: 0.09
Nodes (21): A long prompt among running requests, Batch-1 decode bandwidth, Decode bandwidth, Decode bandwidth, Environment, GPU time per op in a decode step, GPU time per op in a decode step, Launch gaps per forward step (+13 more)

### Community 175 - "Scheduler (Branch 19)"
Cohesion: 0.50
Nodes (4): Continuous batching, Requests, Scheduler (Branch 19), Testing the scheduler

### Community 177 - "write_kv Implementation (kernel/write-kv-impl)"
Cohesion: 0.50
Nodes (4): Kernel indexing, Paged cache writes, Testing, write_kv Implementation (kernel/write-kv-impl)

### Community 178 - "inkSegments"
Cohesion: 0.14
Nodes (29): approach(), block(), buildStrip(), createChunkedPrefillAboutModel(), dashed(), DECODE_AT(), dispose(), Pen (+21 more)

### Community 179 - "BlockManager"
Cohesion: 0.09
Nodes (18): kv/block-manager, kv/block-manager-impl, sched/scheduler-impl, C7. Continuous batching scheduler (week 5), Admission and preemption, Errors, Retiring, Scheduler Implementation (sched/scheduler-impl) (+10 more)

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

### Community 187 - "paged-kv/figures.tsx"
Cohesion: 0.10
Nodes (28): web/continuous-batching-prologue, blocksFor(), continuousBatching(), empty(), finish(), Grid, lengths, slots (+20 more)

### Community 188 - "TEST"
Cohesion: 0.07
Nodes (13): TokenEvent, completion_tokens, finish, step, text, ScriptedBackend, delay_, next_ (+5 more)

### Community 189 - "TEST"
Cohesion: 0.10
Nodes (6): give_words(), make(), next_token(), run_step(), solo(), TEST()

### Community 190 - "Docs (Branch 8)"
Cohesion: 0.50
Nodes (4): Docs (Branch 8), Ownership, Project overview, README

### Community 191 - "gemm/sandbox.tsx"
Cohesion: 0.11
Nodes (26): web/kernel-add-rmsnorm, Branch 55 `web/kernel-add-rmsnorm` (web worktree), from origin/main c3e86c3 (#61 merged), not pushed, GemmOverlay(), EasedBar(), bkNames, Button(), fmt(), kb() (+18 more)

### Community 192 - "Paged Decode Attention (Branch 17)"
Cohesion: 0.67
Nodes (3): Paged attention, Paged Decode Attention (Branch 17), Testing attention

### Community 193 - "createCard"
Cohesion: 0.20
Nodes (26): bytesExhibit(), createFoundationsModel(), FOUNDATIONS_STOP, FoundationsNumbers, fusionExhibit(), hierarchyExhibit(), lines(), memoryExhibit() (+18 more)

### Community 196 - "bench"
Cohesion: 0.11
Nodes (17): tools/llamacpp-baseline, Conventions, CORS, Engine API contract, `GET /health`, `GET /v1/steps` (Server-Sent Events), Host header, HTTP endpoints (`engine serve`) (+9 more)

### Community 198 - "Options"
Cohesion: 0.15
Nodes (15): allowed_hosts(), http_options(), is_loopback(), joined(), model_name(), Options, allow_hosts, cors_origins (+7 more)

### Community 199 - "TEST"
Cohesion: 0.08
Nodes (7): Branch 4: model/config, Graphs for Split Attention (Branch 43), decode_graph_buckets(), bucket_for(), contains(), powers_of_two(), TEST()

### Community 202 - "Part 1 (MVP): component specs"
Cohesion: 0.29
Nodes (7): C1. Core infrastructure (week 1), C2. Weight loader (week 1), C4. KV block manager (week 4), C5. Model runner (weeks 2 to 5), C6. CUDA Graphs (week 6), C8. Sampler (weeks 3 and 5), Part 1 (MVP): component specs

### Community 203 - "Pending"
Cohesion: 0.12
Nodes (16): Pending, arrived, events, id, submission, Submission, max_tokens, params (+8 more)

### Community 204 - "Feed"
Cohesion: 0.13
Nodes (3): Feed, ReplayPlayer, TileValues

### Community 205 - "cellNumber"
Cohesion: 0.20
Nodes (11): stepOps, share, stepOp, stepOp, largestGap(), loadCell(), loadSeries(), roofCol() (+3 more)

### Community 206 - "compilerOptions"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+10 more)

### Community 207 - "three"
Cohesion: 0.07
Nodes (27): web/kernels-data, 53 web/kernels-data (2 commits), 54 web/kernels-template (4 commits), Branches 53 `web/kernels-data` and 54 `web/kernels-template` (web worktree), stacked on web/overview (merged as #60, so a PR against main shows only these 6 commits), not pushed, eslint, eslint-config-next, @fontsource/instrument-serif, @fontsource/jetbrains-mono (+19 more)

### Community 208 - "MetricsWindow"
Cohesion: 0.10
Nodes (12): `GET /metrics`, Finished, time_s, tpot_s, ttft_s, MetricsWindow, finished_, tokens_ (+4 more)

### Community 209 - "app/layout.tsx"
Cohesion: 0.07
Nodes (31): Build and CMake, CI, Compilers, cuBLAS and CUDA facts used, Environment, Forward pass vs HF (branch 13, measured Oct 9), HuggingFace BF16 op arithmetic (transformers 5.17.0, torch 2.11.0+cu128, verified by the branch 11 kernels), HuggingFace model facts (transformers 5.17.0, torch 2.11.0+cu128, verified on the RTX 5060) (+23 more)

### Community 210 - "gloss.tsx"
Cohesion: 0.10
Nodes (23): react-dom, ease, GlossCard(), GlossContext, Place, placeNear(), PointGloss(), PointTerm (+15 more)

### Community 211 - "run"
Cohesion: 0.12
Nodes (10): HttpServer, impl_, add_serve_command(), run(), Answer, finish_reason, text, generate_answer() (+2 more)

### Community 212 - "chat_api.cpp"
Cohesion: 0.31
Nodes (12): server/http, field(), integer(), number(), parse_chat_request(), parse_messages(), parse_stop(), parse_stream() (+4 more)

### Community 213 - "HttpOptions"
Cohesion: 0.12
Nodes (13): SamplingDefaults, max_tokens, temperature, top_k, top_p, HttpOptions, allowed_hosts, block_size (+5 more)

### Community 214 - "cuda-graphs-model.ts"
Cohesion: 0.17
Nodes (25): captureExhibit(), createCudaGraphsModel(), CudaGraphsBookNumbers, fillBar(), fmt(), gapsExhibit(), launchesExhibit(), Line (+17 more)

### Community 215 - "MetricsSnapshot"
Cohesion: 0.10
Nodes (20): Percentiles, p50, p99, Counters, blocks_free, finished_total, output_total, preemptions_total (+12 more)

### Community 216 - "Web site plan (Part 2)"
Cohesion: 0.15
Nodes (12): Branches (in order), Constraints, Decisions, Decisions since the plan (Joe, 2026-10-25 and 26), Design tokens, Libraries (from pick-ui-library), Motion rules, Pages (+4 more)

### Community 217 - "Kernels section plan"
Cohesion: 0.15
Nodes (12): Branches (in order), Data, Decisions (Joe, 2026-10-25), Extra pages (Joe, 2026-10-31), How the template is built (for the next pages), Index page `/kernels`, Kernels section plan, Page template (every kernel page) (+4 more)

### Community 218 - "1. Project context and decisions"
Cohesion: 0.17
Nodes (11): 1.1 Who and what, 1.2 Hardware and environment, 1.3 Key technical decisions, 1.4 Critique of the original blueprint (why the plan changed), 1.5 How to build it (AI usage policy), 1.6 Resume impact analysis (honest assessment), 1.7 Learning plan (before and during the build), 1.8 Working preferences for Claude on this project (+3 more)

### Community 219 - "load_chat_prompts"
Cohesion: 0.11
Nodes (7): Branch 49 `bench/chat-prompts` (worktree ~/code/llm-inference-engine-api), 2 commits on 0d11053, not pushed, Branch 50 `web/scaffold` (worktree ~/code/llm-inference-engine-web), 3 commits on 0d11053, not pushed, Branches 49 and 50 (PR summaries), load_chat_prompts(), ChatPromptsTest, tok, TEST_F()

### Community 221 - "Core concepts you must understand"
Cohesion: 0.18
Nodes (11): 10. Numerics, 1. How a GPU runs code, 2. The memory hierarchy, 3. The roofline model, 4. The Llama forward pass (what you are implementing), 5. Prefill versus decode, 6. The KV cache and why it is paged, 7. Launch overhead and CUDA Graphs (+3 more)

### Community 222 - "parse_safetensors_header"
Cohesion: 0.12
Nodes (15): loader/safetensors-parser, Amendments since (in order), Clarifications Joe answered in the first session, Commits, Delegation, Original prompt (verbatim), Part A: implement fully (I review and test only), Part B: implement as a draft (I will read and explain every line) (+7 more)

### Community 223 - "Working rules"
Cohesion: 0.18
Nodes (10): Code style, Commits, Golden data, Part C stubs, Repo and branches, Reporting, The gate before every commit, The GPU (+2 more)

### Community 224 - "metrics_window.cpp"
Cohesion: 0.35
Nodes (6): MetricsWindow::add_request(), MetricsWindow::add_tokens(), MetricsWindow::output_per_s(), MetricsWindow::tpot_s(), MetricsWindow::ttft_s(), percentiles()

### Community 225 - "SmoothRange"
Cohesion: 0.14
Nodes (25): web/sandbox-chunked-prefill, web/sandbox-foundations, web/sandbox-gemm, web/sandbox-paged-attention, web/sandbox-perf, web/sandbox-rope, web/sandbox-sampling, web/smooth-sliders (+17 more)

### Community 226 - "dependencies"
Cohesion: 0.06
Nodes (33): dependencies, clsx, liveline, motion, next, next-themes, @number-flow/react, react (+25 more)

### Community 228 - "exercise-card.tsx"
Cohesion: 0.15
Nodes (22): exerciseSteps(), Disclosure(), ease, Commands(), ExerciseCard(), exerciseSpec(), fileLink(), OpenSteps() (+14 more)

### Community 229 - "Branch 73 `server/http` summary (issue #95, engine half)"
Cohesion: 0.22
Nodes (8): Branch 73 `server/http` summary (issue #95, engine half), Commits, Deviations from docs/api.md (all recorded there), Draft PR description, Line-by-line review (Part B), Not verified, Open questions for Joe, Verified end to end (release build, RTX 5060, GPU up)

### Community 230 - ".run_staged"
Cohesion: 0.10
Nodes (19): kernel/m4-stubs, runtime/graphs-split-attention, runtime/wire-attention-split, Branch 22 `kernel/m4-stubs` (M4 Part C stubs for Joe), Decisions I made (tell me if you disagree), Not done on purpose, Validation, What Joe implements (all todo_joe, run `ctest --preset debug -L todo_joe`) (+11 more)

### Community 231 - "load_report.py"
Cohesion: 0.24
Nodes (18): aggregate(), curve_plot(), gap_run(), gaps_markdown(), load_run(), main(), measured_line(), meets_slo() (+10 more)

### Community 232 - "paged-kv-model.ts"
Cohesion: 0.25
Nodes (25): appendExhibit(), batchingExhibit(), boxes(), clamp01(), colors(), createPagedKvModel(), floorLabel(), fmt() (+17 more)

### Community 233 - "Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95)"
Cohesion: 0.25
Nodes (7): Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95), Commits, Draft PR description, Not verified, needs Joe or the real server, Open questions for Joe, Verified, and how, What it is

### Community 234 - "LLM Inference Engine Build Plan"
Cohesion: 0.15
Nodes (12): Benchmark rules, Debugging method when outputs are wrong, Generating reference data, LLM Inference Engine Build Plan, Metric definitions, MVP definition of done, Overview, Profiling workflow (+4 more)

### Community 235 - "BlockManager"
Cohesion: 0.15
Nodes (4): BlockManager, length(), Run, Scheduler

### Community 236 - "rope/sandbox.tsx"
Cohesion: 0.19
Nodes (22): k, q, rotateHalfReference(), angle(), HEAD_DIM, invFreq(), MAX_POS, PAIRS (+14 more)

### Community 238 - "Kit"
Cohesion: 0.13
Nodes (22): BookModel, approach(), column(), createAttentionV2AboutModel(), Pen, approach(), bump(), createGemmAboutModel() (+14 more)

### Community 239 - "Part 2: polish, UI/UX, and enhancements"
Cohesion: 0.29
Nodes (7): Part 2: polish, UI/UX, and enhancements, Priority list, Track 1: Serving, Track 2: UI/UX (the web dashboard), Track 3: Runtime features, Track 4: Hard kernels, Track 5: Engineering polish

### Community 240 - "rope-model.ts"
Cohesion: 0.16
Nodes (21): approach(), COL_X(), createPagedKvAboutModel(), fmt(), OURS, Pen, ROW_Z, smooth() (+13 more)

### Community 241 - "attention/sandbox.tsx"
Cohesion: 0.15
Nodes (23): @number-flow/react, directAttention(), makeProblem(), maxDiff(), mulberry32(), pagedAttention(), Problem, Run (+15 more)

### Community 243 - "row"
Cohesion: 0.32
Nodes (3): RoundTrip, row(), Table

### Community 244 - "add-rmsnorm-model.ts"
Cohesion: 0.20
Nodes (21): AddRmsnormNumbers, arcLine(), createAddRmsnormModel(), firstFusedExhibit(), jobExhibit(), LANES, Line, MASKS (+13 more)

### Community 245 - "washPlane"
Cohesion: 0.27
Nodes (20): blocksExhibit(), box100(), cacheExhibit(), createPagedAttentionModel(), decodeExhibit(), Exhibit, fmt(), homeExhibit() (+12 more)

### Community 246 - "Step 8: Day-1 smoke tests and troubleshooting"
Cohesion: 0.40
Nodes (5): 8.1 Smoke test checklist, 8.2 Troubleshooting, 8.3 Where to go next, Sources, Step 8: Day-1 smoke tests and troubleshooting

### Community 250 - "Web site"
Cohesion: 0.40
Nodes (4): Deploy on Vercel, Live mode from the deployed site, Run it, Web site

### Community 251 - "KV Block Manager (Branch 15)"
Cohesion: 0.50
Nodes (4): Interface and errors, KV Block Manager (Branch 15), Paged KV cache, Tests

### Community 252 - "utility"
Cohesion: 0.09
Nodes (4): Event, event, StopMatcher::flush(), TEST()

### Community 256 - "Driver"
Cohesion: 0.13
Nodes (9): Driver, cache_, done_, held_, progress_, expect_chunks(), give(), make() (+1 more)

### Community 270 - "add-rmsnorm-about-model.ts"
Cohesion: 0.15
Nodes (18): A, AddRmsnormAboutNumbers, approach(), Bar, bump(), createAddRmsnormAboutModel(), M, normed() (+10 more)

### Community 271 - "add-rmsnorm/about.tsx"
Cohesion: 0.13
Nodes (18): A, addRmsnormAbout, addRmsnormAboutCaption, AddRmsnormAboutFigure(), Bars(), blockX, hidden, M (+10 more)

### Community 272 - "TEST"
Cohesion: 0.12
Nodes (5): tiny_config(), tinyllama_config(), copy_of(), expect_rejected(), TEST()

### Community 273 - "expect_invariants"
Cohesion: 0.20
Nodes (11): blocks_for(), expect_invariants(), expect_step(), give(), make(), run_to_completion(), RunResult, finished (+3 more)

### Community 274 - "scheduler.cpp"
Cohesion: 0.12
Nodes (7): validate(), Scheduler::cancel(), Scheduler::schedule(), Scheduler::schedule_chunked(), Scheduler::submit(), Scheduler::take_finished(), Scheduler::update()

### Community 275 - "paged-attention-about-model.ts"
Cohesion: 0.20
Nodes (16): approach(), blockX(), bump(), column(), createPagedAttentionAboutModel(), frame(), Pen, slotX() (+8 more)

### Community 276 - "swiglu-model.ts"
Cohesion: 0.34
Nodes (15): bitsExhibit(), createSwigluModel(), fmt(), gridExhibit(), jobExhibit(), lines(), placed(), rail() (+7 more)

### Community 278 - "ChunkedStep"
Cohesion: 0.15
Nodes (11): ChunkedStep, decode, prefill, sampled, PrefillChunk, count, first, id (+3 more)

### Community 279 - "gemm-model.ts"
Cohesion: 0.35
Nodes (14): approach(), batchExhibit(), cellCard(), createGemmModel(), fmt(), GEMM_STOP, gemvExhibit(), jobExhibit() (+6 more)

### Community 280 - "swiglu-about-model.ts"
Cohesion: 0.25
Nodes (14): bar(), createSwigluAboutModel(), fmt(), gateHeight(), gx(), hiddenZ(), num(), outHeight() (+6 more)

### Community 281 - "model_runner.cpp"
Cohesion: 0.21
Nodes (9): Blas::Blas(), gemm(), require(), buffer(), ModelRunner::forward(), ModelRunner::ModelRunner(), require(), rows() (+1 more)

### Community 282 - "prefill_ttft.py"
Cohesion: 0.32
Nodes (12): base_system(), engine_command(), grouped(), llama_command(), llama_ttfts(), main(), measure(), plot_ttft() (+4 more)

### Community 283 - "attention-v2-model.ts"
Cohesion: 0.46
Nodes (12): AttentionV2Numbers, capExhibit(), createAttentionV2Model(), fmt(), lanesExhibit(), onlineExhibit(), placed(), smooth() (+4 more)

### Community 284 - "split-attention-about-model.ts"
Cohesion: 0.26
Nodes (12): approach(), CHOICES, createSplitAboutModel(), fmt(), lerp(), Pen, rangeOf(), smCenter() (+4 more)

### Community 285 - "sample.cu"
Cohesion: 0.39
Nodes (7): block_reduce(), block_sum(), gumbel(), order_key(), sample_kernel(), top_k_threshold(), top_p_threshold()

### Community 286 - "View"
Cohesion: 0.17
Nodes (12): Row-major GEMM through cuBLAS, Legend(), Stats(), GroupLegend, ctx(), n(), profileStats(), Roof() (+4 more)

### Community 287 - "foundations-about-model.ts"
Cohesion: 0.23
Nodes (11): approach(), BARS, createFoundationsAboutModel(), EMB, fmt(), inputOf(), LM, Pen (+3 more)

### Community 288 - "Branch 84: bench/chunked-prefill (issue #101, Part A)"
Cohesion: 0.20
Nodes (9): Branch 84: bench/chunked-prefill (issue #101, Part A), Commits, Draft PR description, Honest reading, Measured (RTX 5060 Laptop GPU, release build, 2026-11-05, commit 1693c46, engine code = main 2eb2f91), Not verified, Open questions for Joe, Verification (+1 more)

### Community 289 - "Branch 98: web/sandbox-perf"
Cohesion: 0.20
Nodes (9): Branch 98: web/sandbox-perf, Commits (first parent, oldest first), Conclusion, Draft PR description, Layout shift (every frame of a glide and of a drag; all rows, cells, separators, the section and the body), Measurements, Not verified, Open questions for Joe (+1 more)

### Community 290 - "Branch 90: tools/onchip-speeds"
Cohesion: 0.25
Nodes (7): Branch 90: tools/onchip-speeds, Caveats, Commits, Gates, Measured (RTX 5060 Laptop GPU, commit e608146, 2026-11-05T04:40Z, 5 runs, median (min to max), SM clock from clock64 vs %globaltimer inside each kernel), Not verified, Open questions

### Community 291 - "FakeRunner"
Cohesion: 0.25
Nodes (5): FakeModel, FakeRunner, block_size_, cache_, model_

### Community 293 - "scoreboard.tsx"
Cohesion: 0.38
Nodes (6): BarGauge(), CellGauge(), Scoreboard(), easeInOut, Criterion, Gauge

### Community 294 - "Step 1: Windows host prep (you, ~45 min)"
Cohesion: 0.33
Nodes (6): 1.1 Update Windows and the NVIDIA driver, 1.2 Install WSL2 and Ubuntu 24.04, 1.3 Configure WSL2 resources and networking, 1.4 Install the editor and terminal, 1.5 Rules for working in WSL2, Step 1: Windows host prep (you, ~45 min)

### Community 295 - "Exercise: sum reduction of 16M floats"
Cohesion: 0.33
Nodes (5): Exercise: sum reduction of 16M floats, Notes and caveats, Results, Setup, Why the three differ

### Community 296 - "Step 5: Partner's Mac setup and remote GPU access (~1.5 hours)"
Cohesion: 0.40
Nodes (5): 5.1 Native toolchain on the Mac, 5.2 The CPU-only build, 5.3 Remote access to your GPU (you set up, partner uses), 5.4 Fallback: Google Colab, Step 5: Partner's Mac setup and remote GPU access (~1.5 hours)

### Community 297 - "Step 7: CI and team conventions (together, ~45 min)"
Cohesion: 0.40
Nodes (5): 7.1 `.github/workflows/ci.yml`, 7.2 Formatting before every commit, 7.3 Pull request template, 7.4 Issues, milestones, and branches, Step 7: CI and team conventions (together, ~45 min)

### Community 298 - "rope_kernel"
Cohesion: 0.40
Nodes (3): InvFreq, v, rope_kernel()

### Community 299 - "Part 1 (MVP): schedule and milestones"
Cohesion: 0.50
Nodes (4): If you fall behind, Milestones, Part 1 (MVP): schedule and milestones, Week-by-week plan

### Community 300 - "Resume bullets, README, and interview prep"
Cohesion: 0.50
Nodes (4): Questions you must be able to answer cold, README structure, Resume bullets, README, and interview prep, Resume entry (Jake's template)

### Community 301 - "On-chip Memory Speeds (Branch 90)"
Cohesion: 0.50
Nodes (4): On-chip Memory Speeds (Branch 90), On the site, The microbenchmark, The numbers

## Knowledge Gaps
- **526 isolated node(s):** `id`, `prompt_tokens`, `output_tokens`, `preemptions`, `arrival_s` (+521 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 3163 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **73 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Changelog` connect `Changelog` to `Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)`, `serve_step`, `chapter.ts`, `ModelRunner`, `build-timeline.tsx`, `Profiling`, `scene-figures.tsx`, `paged_attention_harness.h`, `ModelWeights`, `PagedRunner`, `add_rmsnorm.cu`, `SafetensorsHeader`, `argparse`, `version`, `test_swiglu_fast.cpp`, `front-matter.ts`, `Scheduler`, `ModelConfig`, `write_kv`, `sync-data.mjs`, `story-view.tsx`, `BlockManager`, `Status`, `compare_bf16`, `paged-kv/figures.tsx`, `gemm/sandbox.tsx`, `WorkloadConfig`, `bench`, `ReferencePrompt`, `three`, `book-scene.ts`, `chat_api.cpp`, `parse_safetensors_header`, `SmoothRange`, `DType`, `.run_staged`?**
  _High betweenness centrality (0.268) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `useLoop()` (e.g. with `web/kernel-add-rmsnorm` and `web/kernels-template`) actually correct?**
  _`useLoop()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `id`, `prompt_tokens`, `output_tokens` to the rest of the system?**
  _526 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Merged to main (#6 to #13 squash-merged; from #14 on, rebase and merge)` be split into smaller, more focused modules?**
  _Cohesion score 0.0957983193277311 - nodes in this community are weakly interconnected._
- **Why does `web/kernels-template` connect `story-view.tsx` to `reading-view.tsx`, `.run_staged`, `useLoop`, `build-timeline.tsx`, `Changelog`, `app/layout.tsx`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Should `chapter.ts` be split into smaller, more focused modules?**
  _Cohesion score 0.03312392918332382 - nodes in this community are weakly interconnected._
- **Why does `useLoop()` connect `useLoop` to `sampling/story.tsx`, `add-rmsnorm/story.tsx`, `add-rmsnorm/about.tsx`, `measured`, `scene-figures.tsx`, `usePrefersReducedMotion`, `sampling/about.tsx`, `story-view.tsx`, `paged-kv/figures.tsx`, `gpu-floorplan.tsx`, `foundations/story.tsx`, `cuda-graphs/story.tsx`, `gemm/sandbox.tsx`, `Label`, `rope/story.tsx`, `paged-kv/story.tsx`, `cuda-graphs-model.ts`, `split-attention/story.tsx`, `attention-v2/story.tsx`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._