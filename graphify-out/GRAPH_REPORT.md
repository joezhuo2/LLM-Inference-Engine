# Graph Report - llm-inference-engine  (2026-10-25)

## Corpus Check
- 232 files · ~135,483 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 94 file(s) not represented in the graph (top: .csv 69, (none) 25)

## Summary
- 2599 nodes · 5371 edges · 202 communities (149 shown, 53 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 670 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1172a22c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- TensorChecksum
- TEST
- tinyllama_advanced.py
- cstddef
- ModelRunner
- cstdint
- sample.cu
- Part 0 - Foundations
- TEST
- .data
- TEST
- SafetensorsFile
- TEST
- WeightLayout
- TEST
- TEST
- sys
- MappedFile
- Request
- TEST_F
- from_bf16
- test_add_rmsnorm.cpp
- TEST
- ModelWeights
- TEST
- TEST
- PagedRunner
- TEST
- plan_weight_layout
- Tensor
- load_weights
- Options
- TEST
- baseline_llamacpp.py
- TEST
- Scheduler
- ModelConfig
- TEST
- write_kv
- RequestMetrics
- Summary
- test_rope.cpp
- fail
- TEST
- PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)
- BlockManager
- GemmTest
- mapped_file.cpp
- kv/block-manager
- TextStreamTest
- block_manager.cpp
- TEST
- run
- TEST_F
- WeightUploadTest
- build_paged_batch
- TEST
- Options
- TEST
- cpu_attention
- SafetensorsHeader
- make_workload
- WorkloadConfig
- Tokenizer
- llm-inference-engine
- config.cpp
- ReferencePrompt
- scheduler.cpp
- PinnedBuffer
- Batch
- Part 1 - Engine (M1 Foundations)
- expected_tensors
- TEST
- SamplingParams
- AddTester.cu
- vector
- parse_safetensors_header
- TEST_F
- from_file
- bandwith_test.cu
- golden_dir
- .kv_dim
- TEST
- math
- TEST_F
- CudaStream
- TEST
- TEST_P
- TEST
- Naive Oracle Kernels (Branch 11)
- Forward Pass (Branch 13)
- paged_attention_decode Implementation (kernel/paged-attention-impl)
- LengthRange
- TEST
- ChatMessage
- ModelTest
- DType
- main.cpp
- Batched Sampling (Branch 18)
- Tokenizer (Branch 10)
- Weight Upload (Branch 6)
- Sampling (Branch 14)
- Exercise: sum reduction of 16M floats
- model/forward
- PagedBatch
- iostream
- run
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
- test_smoke.cpp
- roofline.py
- rope_kernel
- TEST
- Build Scaffolding (Branch 1)
- add.cu
- Safetensors Loader Stub (Branch 5)
- request_metrics
- profile_report.py
- serve_step
- Shape
- ServeTest
- TEST
- TEST_F
- Event
- TEST
- TEST
- TEST
- TEST
- TEST
- TEST
- TEST
- TEST
- attention_kernel
- embedding_kernel
- Changelog
- rmsnorm_kernel
- swiglu_kernel
- tensor.h
- Shape
- bandwith_test.md
- dram_throughput.md
- format.sh
- paged_attention_harness.h
- Blocks
- TEST_F
- Blas
- add_rmsnorm.cu
- TEST
- as_bytes
- run_kernel
- plot.py
- Baselines and plots
- swiglu_fast.cu
- test_baseline_hf.py
- reference
- compare_bf16
- SampleRow
- DecodeGraphs
- decode_step_b16.md
- Plot
- TempFile
- Safe Tensor Reader (Task 2)
- Results
- Scheduler (Branch 19)
- test_profile_report.py
- write_kv Implementation (kernel/write-kv-impl)
- Buffer (Task 1)
- PagedRunner::run_staged
- paged_attention_split_merge_kernel
- reproduce.sh
- test_baseline_llamacpp.py
- generate
- decode_step.md
- decode_step_graphs.md
- rope_kernel
- read
- Arrival
- parse
- Docs (Branch 8)
- OwnedStream
- Paged Decode Attention (Branch 17)
- Shape
- setup_llamacpp.sh
- profile.sh
- paged_attention_decode_v2_kernel
- write_kv_kernel
- PagedRunner::PagedRunner
- Repo Housekeeping (Branch 9)

## God Nodes (most connected - your core abstractions)
1. `Tensor` - 69 edges
2. `Part 1 - Engine (M1 Foundations)` - 58 edges
3. `Changelog` - 51 edges
4. `PagedRunner` - 44 edges
5. `TEST()` - 41 edges
6. `Request` - 38 edges
7. `TEST()` - 38 edges
8. `ModelConfig` - 37 edges
9. `ModelRunner` - 36 edges
10. `TEST()` - 36 edges

## Surprising Connections (you probably didn't know these)
- `Reading these numbers` --references--> `ModelRunner`  [INFERRED]
  docs/results.md → include/engine/runtime/model_runner.h
- `Decode bandwidth` --references--> `paged_attention_decode_v2()`  [INFERRED]
  docs/results.md → src/kernels/paged_attention_v2.cu
- `Paged attention` --references--> `write_kv()`  [INFERRED]
  docs/exercises/learning_log.md → src/kernels/write_kv.cu
- `Paged cache writes` --references--> `write_kv()`  [INFERRED]
  docs/exercises/learning_log.md → src/kernels/write_kv.cu
- `The three parts` --references--> `write_kv()`  [INFERRED]
  docs/ownership.md → src/kernels/write_kv.cu

## Import Cycles
- None detected.

## Communities (202 total, 53 thin omitted)

### Community 0 - "TensorChecksum"
Cohesion: 0.15
Nodes (16): tools/checksum, dtype_name(), TensorChecksum, abs_sum, dtype, name, shape, sum (+8 more)

### Community 1 - "TEST"
Cohesion: 0.11
Nodes (4): check(), run_paged(), TEST(), single_row()

### Community 2 - "tinyllama_advanced.py"
Cohesion: 0.11
Nodes (23): capture(), keep(), chat_ids(), dump_prompt(), main(), fuzz_strings(), main(), apply_rope() (+15 more)

### Community 4 - "ModelRunner"
Cohesion: 0.11
Nodes (17): runtime/wire-fast-kernels, ModelRunner, act_, attn_, blas_, config_, gate_up_, kv_ (+9 more)

### Community 6 - "sample.cu"
Cohesion: 0.19
Nodes (11): sampling/batched, better(), block_argmax(), greedy_kernel(), block_reduce(), block_sum(), gumbel(), order_key() (+3 more)

### Community 7 - "Part 0 - Foundations"
Cohesion: 0.14
Nodes (13): Atomics and shared memory, Bandwith Test (Task 3/CUDA Task 1), Core C++, Core Ideas, Core Ideas, Execution model, Foundational, Memory (+5 more)

### Community 11 - ".data"
Cohesion: 0.10
Nodes (11): DeviceBuffer, ptr_, DeviceBuffer::DeviceBuffer(), TEST(), TEST(), download(), upload(), split_kernel() (+3 more)

### Community 13 - "SafetensorsFile"
Cohesion: 0.14
Nodes (5): cpu_add(), TEST(), SafetensorsFile, file_, header_

### Community 14 - "TEST"
Cohesion: 0.12
Nodes (10): Case, block_size, head_dim, heads, kv_heads, num_blocks, check(), random_blocks() (+2 more)

### Community 15 - "WeightLayout"
Cohesion: 0.08
Nodes (27): CopyOp, bytes, dst, name, shape, src, LayerLayout, ln1 (+19 more)

### Community 16 - "TEST"
Cohesion: 0.06
Nodes (18): sched/scheduler, blocks_for(), expect_invariants(), expect_step(), FakeModel, FakeRunner, block_size_, cache_ (+10 more)

### Community 17 - "TEST"
Cohesion: 0.14
Nodes (9): Cache, l_, rows_, Layout, head_dim, heads, kv_heads, outside_value_scale() (+1 more)

### Community 19 - "sys"
Cohesion: 0.06
Nodes (9): Metrics, Workload, Correctness, row(), Scoreboard, write_summary(), MainTest, ProfileTest (+1 more)

### Community 20 - "MappedFile"
Cohesion: 0.15
Nodes (5): MappedFile, data_, size_, TEST(), TEST()

### Community 21 - "Request"
Cohesion: 0.07
Nodes (21): Workload, arrivals, requests, Request, arrived, finished, first_token, id (+13 more)

### Community 22 - "TEST_F"
Cohesion: 0.18
Nodes (3): GenerateTest, stream, TEST_F()

### Community 23 - "from_bf16"
Cohesion: 0.14
Nodes (9): ModelRunnerTest, stream, TEST_F(), to_host(), from_bf16(), argmax_rows(), count_equal(), mean_abs_diff() (+1 more)

### Community 24 - "test_add_rmsnorm.cpp"
Cohesion: 0.18
Nodes (13): AddRmsnormShapeTest, cpu_add(), cpu_rmsnorm(), Result, norm, sum, run(), Shape (+5 more)

### Community 26 - "ModelWeights"
Cohesion: 0.10
Nodes (18): loader/weight-upload, Components so far (M1), Ownership, The three parts, LayerWeights, ln1, ln2, wdown (+10 more)

### Community 27 - "TEST"
Cohesion: 0.13
Nodes (4): expect_rejected(), TEST(), with(), without()

### Community 28 - "TEST"
Cohesion: 0.15
Nodes (7): KvCache, buffer_, head_dim_, kv_heads_, small(), TEST(), tinyllama()

### Community 29 - "PagedRunner"
Cohesion: 0.09
Nodes (21): PagedRunner, act_, attn_, blas_, cache_, config_, gate_up_, ints_ (+13 more)

### Community 30 - "TEST"
Cohesion: 0.15
Nodes (4): Buffer, memory_address, requested_size, TEST()

### Community 31 - "plan_weight_layout"
Cohesion: 0.32
Nodes (7): bf16_bytes(), fail(), plan_weight_layout(), Planner, copies_, used_, shape_string()

### Community 32 - "Tensor"
Cohesion: 0.10
Nodes (17): Tensor, dtype, kMaxDims, ndim, shape, gemm(), require(), rope() (+9 more)

### Community 33 - "load_weights"
Cohesion: 0.12
Nodes (11): DeviceWeights, buffer, config, layout, stats, weights, UploadStats, bytes (+3 more)

### Community 34 - "Options"
Cohesion: 0.10
Nodes (18): open_csv(), Options, graphs, max_batch, model, num_requests, out, output_len (+10 more)

### Community 36 - "baseline_llamacpp.py"
Cohesion: 0.28
Nodes (13): batched_command(), batched_main(), bench_main(), gpu_name(), main(), parse_batched_bench(), parse_llama_bench(), run_llama_bench() (+5 more)

### Community 37 - "TEST"
Cohesion: 0.13
Nodes (3): copy_of(), expect_rejected(), TEST()

### Community 38 - "Scheduler"
Cohesion: 0.14
Nodes (12): Scheduler, block_manager_, config_, finished_list_, pending_step_, pending_step_obj_, requests_, running_list_ (+4 more)

### Community 39 - "ModelConfig"
Cohesion: 0.13
Nodes (16): ModelConfig, bos_id, eos_id, head_dim, heads, hidden, intermediate, kv_heads (+8 more)

### Community 40 - "TEST"
Cohesion: 0.14
Nodes (3): bf16_bytes(), bytes_of(), TEST()

### Community 41 - "write_kv"
Cohesion: 0.13
Nodes (16): kernel/m4-stubs, kernel/naive-attention, kernel/write-kv-impl, kv/paged-cache, runtime/wire-attention-split, sampling/temperature, tools/profiling, Timeline and report (Nsight Systems) (+8 more)

### Community 42 - "RequestMetrics"
Cohesion: 0.15
Nodes (12): RequestMetrics, arrival_s, id, latency_s, output_tokens, preemptions, prompt_tokens, tpot_s (+4 more)

### Community 43 - "Summary"
Cohesion: 0.13
Nodes (15): Stat, mean, p50, p90, p99, Summary, latency_s, output_throughput (+7 more)

### Community 44 - "test_rope.cpp"
Cohesion: 0.16
Nodes (10): cpu_rope(), gpu_rope(), Layout, head_dim, heads, kv_heads, RopeLayoutTest, spread_positions() (+2 more)

### Community 45 - "fail"
Cohesion: 0.20
Nodes (5): fail(), Tokenizer::append_text(), Tokenizer::decode(), Tokenizer::encode(), Tokenizer::Tokenizer()

### Community 46 - "TEST"
Cohesion: 0.18
Nodes (6): kv_block_bytes(), kv_cache_blocks(), KvCache::KvCache(), TEST(), tiny(), tinyllama()

### Community 47 - "PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)"
Cohesion: 0.14
Nodes (12): Attention, Core Ideas, Generation loop, KV cache, MLP, Model shapes, Normalization, Numerics (+4 more)

### Community 48 - "BlockManager"
Cohesion: 0.17
Nodes (10): Host code and measurement, BlockManager, block_size_, free_blocks_, free_count_, num_blocks_, sequences_, Sequence (+2 more)

### Community 49 - "GemmTest"
Cohesion: 0.12
Nodes (7): GemmShapeTest, GemmTest, blas_, stream_, workspace_, TEST_F(), TEST_P()

### Community 50 - "mapped_file.cpp"
Cohesion: 0.23
Nodes (4): fail(), Fd, fd, MappedFile::MappedFile()

### Community 51 - "kv/block-manager"
Cohesion: 0.29
Nodes (4): kv/block-manager, kv/block-manager-impl, BlockManager::allocate(), BlockManager::append_slot()

### Community 52 - "TextStreamTest"
Cohesion: 0.18
Nodes (3): TEST_F(), TextStreamTest, tok

### Community 53 - "block_manager.cpp"
Cohesion: 0.19
Nodes (4): BlockManager::can_append(), BlockManager::free(), BlockManager::length(), BlockManager::slot()

### Community 55 - "run"
Cohesion: 0.16
Nodes (11): add_bench_command(), format_range(), gpu_name(), graph_buckets(), run(), utc_now(), csv_escape(), csv_number() (+3 more)

### Community 56 - "TEST_F"
Cohesion: 0.18
Nodes (3): BenchTest, stream, TEST_F()

### Community 57 - "WeightUploadTest"
Cohesion: 0.17
Nodes (5): TEST_P(), WeightUploadTest, file_, layout_, stream_

### Community 58 - "build_paged_batch"
Cohesion: 0.16
Nodes (8): build_paged_batch(), require(), blocks_for(), expect_consistent(), Snapshot, free, lengths, tables

### Community 59 - "TEST"
Cohesion: 0.14
Nodes (7): kernel/add-rmsnorm-impl, rmsnorm(), cpu_rmsnorm(), gpu_rmsnorm(), RmsnormShapeTest, TEST(), TEST_P()

### Community 60 - "Options"
Cohesion: 0.17
Nodes (9): Options, max_new_tokens, model, prompt, seed, system, temperature, top_k (+1 more)

### Community 61 - "TEST"
Cohesion: 0.11
Nodes (4): attention_splits(), sample_rows(), make(), TEST()

### Community 62 - "cpu_attention"
Cohesion: 0.17
Nodes (10): data, attention_scale(), AttentionLayoutTest, cpu_attention(), Expected, out, slack, outside_softmax_rounding() (+2 more)

### Community 63 - "SafetensorsHeader"
Cohesion: 0.12
Nodes (11): loader/safetensors-stub, SafetensorsHeader, data_offset, tensors, TensorEntry, bytes, dtype, offset (+3 more)

### Community 64 - "make_workload"
Cohesion: 0.31
Nodes (3): make_workload(), SplitMix64, state_

### Community 65 - "WorkloadConfig"
Cohesion: 0.18
Nodes (11): WorkloadConfig, bos_id, eos_id, num_requests, output_len, params, prompt_len, request_rate (+3 more)

### Community 66 - "Tokenizer"
Cohesion: 0.12
Nodes (12): tools/bench, TextStream, emitted_, ids_, tokenizer_, SentencePieceProcessor, Tokenizer, sentinel_ (+4 more)

### Community 67 - "llm-inference-engine"
Cohesion: 0.18
Nodes (10): Architecture, Beyond the MVP, Building, Design decisions, Goals, Hardware, llm-inference-engine, Ownership (+2 more)

### Community 68 - "config.cpp"
Cohesion: 0.33
Nodes (5): fail(), field(), load_config(), parse_config(), require_if_present()

### Community 69 - "ReferencePrompt"
Cohesion: 0.17
Nodes (13): tokenizer/sentencepiece, load_reference_manifest(), reference_dir(), reference_skip_reason(), ReferencePrompt, file, generated_ids, messages (+5 more)

### Community 70 - "scheduler.cpp"
Cohesion: 0.12
Nodes (12): SchedulerConfig, block_size, eos_id, max_batch, max_context, max_prefill_tokens, num_blocks, Scheduler::schedule() (+4 more)

### Community 71 - "PinnedBuffer"
Cohesion: 0.22
Nodes (4): PinnedBuffer, ptr_, PinnedBuffer::PinnedBuffer(), TEST()

### Community 72 - "Batch"
Cohesion: 0.36
Nodes (4): Batch, sampler_, stream_, vocab_

### Community 73 - "Part 1 - Engine (M1 Foundations)"
Cohesion: 0.06
Nodes (35): Add RMSNorm Tuning (Branch 39), Baselines (Branch 29), Batched Benchmarking (Branch 23), CI Hardening (build/ci-timeouts), Decode CUDA Graphs (Branch 22), Fast RoPE (Branch 22), Fast RoPE (Branch 31), Fast SwiGLU (Branch 22) (+27 more)

### Community 74 - "expected_tensors"
Cohesion: 0.29
Nodes (4): Expected, dtype, shape, expected_tensors()

### Community 75 - "TEST"
Cohesion: 0.20
Nodes (3): cpu_greedy(), run_greedy(), TEST()

### Community 76 - "SamplingParams"
Cohesion: 0.22
Nodes (8): SamplingParams, seed, temperature, top_k, top_p, validate(), drawn(), histogram()

### Community 77 - "AddTester.cu"
Cohesion: 0.14
Nodes (8): tools/llamacpp-baseline, Running, sumNaiveAtomics(), sumSharedMemory(), bench(), main(), sumWarpShuffle(), warpReduceSum()

### Community 78 - "vector"
Cohesion: 0.24
Nodes (8): bf16_row(), repeat_rows(), softmax(), truncate(), Truncated, margin, p, truncated_softmax()

### Community 79 - "parse_safetensors_header"
Cohesion: 0.28
Nodes (5): loader/safetensors-parser, get_dtype_size(), parse_safetensors_header(), throw_error(), throw_tensor_error()

### Community 80 - "TEST_F"
Cohesion: 0.14
Nodes (3): TEST_F(), TokenizerTest, tok

### Community 81 - "from_file"
Cohesion: 0.21
Nodes (9): add_checksum_command(), from_device(), from_file(), Options, from_file, model, out, run() (+1 more)

### Community 82 - "bandwith_test.cu"
Cohesion: 0.36
Nodes (6): launch(), main(), timeKernel(), upload(), vectorAdd(), verify()

### Community 83 - "golden_dir"
Cohesion: 0.40
Nodes (3): golden_checksums_path(), golden_dir(), model_dir()

### Community 84 - ".kv_dim"
Cohesion: 0.36
Nodes (5): model/config, Config semantics, Model Config (Branch 4), Parsing, Testing

### Community 86 - "math"
Cohesion: 0.17
Nodes (15): greedy_batch(), main(), run(), summary_from_batched_row(), bandwidth_fields(), _number(), percentile(), request_metrics() (+7 more)

### Community 87 - "TEST_F"
Cohesion: 0.11
Nodes (11): bucket_of(), prompt_of(), Rig, cache, runner, scheduler, vocab, run_to_completion() (+3 more)

### Community 88 - "CudaStream"
Cohesion: 0.22
Nodes (6): Decode CUDA Graphs (Branch 34), Split attention in CUDA Graphs (Branch 43), DecodeGraphsTest, stream, CudaStream, stream_

### Community 90 - "TEST_P"
Cohesion: 0.08
Nodes (3): PagedLayoutSplitTest, PagedSplitTest, TEST_P()

### Community 92 - "Naive Oracle Kernels (Branch 11)"
Cohesion: 0.29
Nodes (7): Embedding, Naive Oracle Kernels (Branch 11), Residual add, RMSNorm, RoPE kernel, SwiGLU kernel, Testing kernels

### Community 93 - "Forward Pass (Branch 13)"
Cohesion: 0.25
Nodes (7): engine generate, Forward Pass (Branch 13), Free-running generation, Golden harness, Greedy decoding, Model runner, Streaming text

### Community 94 - "paged_attention_decode Implementation (kernel/paged-attention-impl)"
Cohesion: 0.29
Nodes (7): Host side, Kernel structure, Numerics, paged_attention_decode Implementation (kernel/paged-attention-impl), Shared memory, Softmax and reductions, Testing

### Community 95 - "LengthRange"
Cohesion: 0.33
Nodes (6): LengthRange, max, min, check(), parse_length_range(), parse_positive()

### Community 96 - "TEST"
Cohesion: 0.20
Nodes (5): cpu_swiglu(), gpu_swiglu(), TEST(), round_bf16(), to_bf16()

### Community 97 - "ChatMessage"
Cohesion: 0.29
Nodes (4): ChatMessage, content, role, chat_prompt()

### Community 99 - "DType"
Cohesion: 0.10
Nodes (14): core/tensor-and-buffers, CUstream_st, DType, BF16, F16, F32, I32, I64 (+6 more)

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

### Community 105 - "Exercise: sum reduction of 16M floats"
Cohesion: 0.33
Nodes (5): Exercise: sum reduction of 16M floats, Notes and caveats, Results, Setup, Why the three differ

### Community 106 - "model/forward"
Cohesion: 0.29
Nodes (3): model/forward, TEST(), to_host()

### Community 107 - "PagedBatch"
Cohesion: 0.15
Nodes (13): PagedBatch, block_tables, context_lens, logit_rows, max_blocks, num_seqs, positions, prefill (+5 more)

### Community 109 - "run"
Cohesion: 0.29
Nodes (3): add_generate_command(), run(), TEST()

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
Cohesion: 0.40
Nodes (5): cuBLAS GEMM (Branch 3), cuBLAS setup, Numerics, Row-major GEMM through cuBLAS, Testing

### Community 119 - "pull_request_template.md"
Cohesion: 0.40
Nodes (4): How it was tested, Performance (if relevant), Screenshots (ncu / nsys, if relevant), What changed

### Community 121 - "roofline.py"
Cohesion: 0.19
Nodes (15): checksums(), main(), gemm_cost(), main(), markdown(), model_shape(), op_costs(), op_points() (+7 more)

### Community 122 - "rope_kernel"
Cohesion: 0.40
Nodes (3): InvFreq, v, rope_kernel()

### Community 123 - "TEST"
Cohesion: 0.12
Nodes (8): check(), consecutive(), Layout, head_dim, heads, kv_heads, random_positions(), TEST()

### Community 124 - "Build Scaffolding (Branch 1)"
Cohesion: 0.50
Nodes (4): Build Scaffolding (Branch 1), CI and tooling, CMake and dependencies, Python

### Community 125 - "add.cu"
Cohesion: 0.33
Nodes (3): add(), add_kernel(), same_shape()

### Community 126 - "Safetensors Loader Stub (Branch 5)"
Cohesion: 0.50
Nodes (4): Mapping files, Safetensors Loader Stub (Branch 5), Safetensors parsing (your Part C), Testing

### Community 128 - "profile_report.py"
Cohesion: 0.15
Nodes (18): Interface and errors, KV Block Manager (Branch 15), Paged KV cache, Tests, attribute(), bandwidth_section(), busy_and_span(), export_sqlite() (+10 more)

### Community 129 - "serve_step"
Cohesion: 0.20
Nodes (13): bench/batched, runtime/cuda-graphs-impl, CUevent_st, Sampler, rows_, ServeStats, min_free_blocks, prefill_steps (+5 more)

### Community 130 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, b, in, out

### Community 131 - "ServeTest"
Cohesion: 0.15
Nodes (4): ServeTest, prompts, stream, TEST_F()

### Community 132 - "TEST"
Cohesion: 0.15
Nodes (5): check(), count_mismatches(), same_bits_or_both_nan(), TEST(), TEST_P()

### Community 133 - "TEST_F"
Cohesion: 0.09
Nodes (12): model/paged-runner, sched/scheduler-impl, Admission and preemption, Errors, Retiring, Scheduler Implementation (sched/scheduler-impl), Structure, PagedRunnerTest (+4 more)

### Community 145 - "Changelog"
Cohesion: 0.06
Nodes (38): build/ci-timeouts, build/repo-housekeeping, Changelog, docs/ignore-claude-notes, docs/m1-readme, kernel/cublas-gemm, kernel/paged-attention-impl, kernel/paged-attention-split-impl (+30 more)

### Community 149 - "Shape"
Cohesion: 0.67
Nodes (3): Shape, hidden, tokens

### Community 154 - "paged_attention_harness.h"
Cohesion: 0.12
Nodes (19): PagedLayoutTest, TEST_P(), PagedLayoutV2Test, TEST_P(), Batch, k, l, lens (+11 more)

### Community 156 - "TEST_F"
Cohesion: 0.16
Nodes (3): BatchedBenchTest, stream, TEST_F()

### Community 157 - "Blas"
Cohesion: 0.39
Nodes (3): Blas, cublasContext, TEST()

### Community 158 - "add_rmsnorm.cu"
Cohesion: 0.23
Nodes (9): kernel/add-rmsnorm-tune, Planned components, add_rmsnorm(), add_rmsnorm_kernel(), add_rmsnorm_vec_kernel(), aligned16(), blockAllReduceSum(), blockReduceSum() (+1 more)

### Community 159 - "TEST"
Cohesion: 0.11
Nodes (7): TEST(), GuardedBf16, begin_, buffer_, kGuard, kSentinel, n_

### Community 161 - "run_kernel"
Cohesion: 0.33
Nodes (6): kernel/paged-attention-v2-stub, check(), run_paged(), check_kernel(), expect_close(), run_kernel()

### Community 162 - "plot.py"
Cohesion: 0.15
Nodes (21): docs/results, best(), environment_block(), main(), median_of(), parse_correctness(), parse_environment(), replace_block() (+13 more)

### Community 163 - "Baselines and plots"
Cohesion: 0.18
Nodes (8): build/m1-scaffolding, Baselines and plots, HuggingFace transformers, llama.cpp, Plots, Reading the comparison fairly, The workload, version()

### Community 164 - "swiglu_fast.cu"
Cohesion: 0.33
Nodes (5): sm_count_of_current_device(), swiglu(), swiglu_kernel(), swiglu_value(), swiglu_vec_kernel()

### Community 166 - "reference"
Cohesion: 0.22
Nodes (9): Expected, out, slack, Layout, block_size, head_dim, heads, kv_heads (+1 more)

### Community 167 - "compare_bf16"
Cohesion: 0.13
Nodes (10): kernel/naive-oracles, bf16_ulps(), Bf16Diff, first, max_ulps, mismatches, close_up_to_reduction_order(), compare_bf16() (+2 more)

### Community 168 - "SampleRow"
Cohesion: 0.33
Nodes (4): SampleRow, params, step, Sampler::operator()()

### Community 169 - "DecodeGraphs"
Cohesion: 0.18
Nodes (5): runtime/graphs-split-attention, DecodeGraphs, buckets_, graphs_, runner_

### Community 170 - "decode_step_b16.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 173 - "Safe Tensor Reader (Task 2)"
Cohesion: 0.25
Nodes (8): Build, Bytes and types, Core Ideas, Errors and program entry, Memory mapping, Opening and inspecting files, Safe Tensor Reader (Task 2), Safetensors format

### Community 174 - "Results"
Cohesion: 0.11
Nodes (17): Batch-1 decode bandwidth, Decode bandwidth, Decode bandwidth, Environment, GPU time per op in a decode step, GPU time per op in a decode step, Launch gaps per forward step, Launch gaps per forward step (+9 more)

### Community 175 - "Scheduler (Branch 19)"
Cohesion: 0.50
Nodes (4): Continuous batching, Requests, Scheduler (Branch 19), Testing the scheduler

### Community 176 - "test_profile_report.py"
Cohesion: 0.14
Nodes (4): Attribution, Bandwidth, BusyAndSpan, make_db()

### Community 177 - "write_kv Implementation (kernel/write-kv-impl)"
Cohesion: 0.50
Nodes (4): Kernel indexing, Paged cache writes, Testing, write_kv Implementation (kernel/write-kv-impl)

### Community 178 - "Buffer (Task 1)"
Cohesion: 0.33
Nodes (6): Buffer (Task 1), Core Ideas, Memory, special member functions, Testing, Values and references

### Community 179 - "PagedRunner::run_staged"
Cohesion: 0.18
Nodes (7): NvtxRange, ranged(), PagedRunner::logits(), PagedRunner::run_staged(), PagedRunner::stage(), require(), view()

### Community 181 - "reproduce.sh"
Cohesion: 0.73
Nodes (5): bench(), graph_bench(), quiet(), reproduce.sh script, usage()

### Community 182 - "test_baseline_llamacpp.py"
Cohesion: 0.13
Nodes (3): batched_row(), LlamaBatchedBench, LlamaBench

### Community 184 - "decode_step.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 185 - "decode_step_graphs.md"
Cohesion: 0.40
Nodes (4): Decode bandwidth, GPU time per op in a decode step, Launch gaps per forward step, Top 12 kernels

### Community 186 - "rope_kernel"
Cohesion: 0.40
Nodes (3): InvFreq, v, rope_kernel()

### Community 188 - "Arrival"
Cohesion: 0.40
Nodes (5): Arrival, at_iteration, id, max_new_tokens, prompt_len

### Community 189 - "parse"
Cohesion: 0.83
Nodes (3): expect_rejected(), expect_rejected_header(), parse()

### Community 190 - "Docs (Branch 8)"
Cohesion: 0.50
Nodes (4): Docs (Branch 8), Ownership, Project overview, README

### Community 192 - "Paged Decode Attention (Branch 17)"
Cohesion: 0.67
Nodes (3): Paged attention, Paged Decode Attention (Branch 17), Testing attention

### Community 193 - "Shape"
Cohesion: 0.50
Nodes (4): Shape, ff, shift, tokens

## Knowledge Gaps
- **348 isolated node(s):** `id`, `prompt_tokens`, `output_tokens`, `preemptions`, `arrival_s` (+343 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1345 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **53 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tensor` connect `Tensor` to `TEST`, `TEST`, `TEST_F`, `.data`, `TEST`, `TEST`, `WeightLayout`, `Changelog`, `stdexcept`, `TEST`, `tensor.h`, `from_bf16`, `ModelWeights`, `add_rmsnorm.cu`, `TEST`, `swiglu_fast.cu`, `SampleRow`, `write_kv`, `PagedRunner::run_staged`, `TEST`, `cpu_attention`, `Batch`, `SamplingParams`, `TEST_F`, `TEST_P`, `DType`, `model/forward`, `PagedBatch`, `TEST`, `TEST`, `add.cu`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `Tensor` (e.g. with `.step()` and `cpu_attention()`) actually correct?**
  _`Tensor` has 13 INFERRED edges - model-reasoned connections that need verification._
- **What connects `id`, `prompt_tokens`, `output_tokens` to the rest of the system?**
  _348 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `TEST` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Why does `Part 1 - Engine (M1 Foundations)` connect `Part 1 - Engine (M1 Foundations)` to `profile_report.py`, `TEST_F`, `Part 0 - Foundations`, `Scheduler (Branch 19)`, `write_kv Implementation (kernel/write-kv-impl)`, `Docs (Branch 8)`, `Paged Decode Attention (Branch 17)`, `Repo Housekeeping (Branch 9)`, `.kv_dim`, `Naive Oracle Kernels (Branch 11)`, `Forward Pass (Branch 13)`, `paged_attention_decode Implementation (kernel/paged-attention-impl)`, `Batched Sampling (Branch 18)`, `Tokenizer (Branch 10)`, `Weight Upload (Branch 6)`, `Sampling (Branch 14)`, `Block Manager Implementation (kv/block-manager-impl)`, `Naive Attention (Branch 12)`, `Bench Harness (Branch 21)`, `Tensor and Buffers (Branch 2)`, `Paged KV Cache (Branch 16)`, `Checksums (Branch 7)`, `cuBLAS GEMM (Branch 3)`, `Build Scaffolding (Branch 1)`, `Safetensors Loader Stub (Branch 5)`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Should `tinyllama_advanced.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11428571428571428 - nodes in this community are weakly interconnected._
- **Why does `bench()` connect `AddTester.cu` to `BlockManager`, `bandwith_test.cu`, `Baselines and plots`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._