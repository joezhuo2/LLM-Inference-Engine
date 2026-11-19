export type EngineCell = { path: string; line: number; symbol: string; note: string } | { none: string };
export type EngineKey = "vllm" | "llamacpp";
export type EngineRow = { slug: string; idea: string } & Record<EngineKey, EngineCell>;

export const enginesTitle = "where this lives in other engines";
export const enginesChecked = "2026-11-18";
export const engines: Record<EngineKey, { name: string; repo: string; tag: string }> = {
  vllm: { name: "vLLM", repo: "https://github.com/vllm-project/vllm", tag: "v0.34.0" },
  llamacpp: { name: "llama.cpp", repo: "https://github.com/ggml-org/llama.cpp", tag: "v1.0.0" },
};
export const engineKeys: EngineKey[] = ["vllm", "llamacpp"];

export const engineHref = (key: EngineKey, cell: EngineCell) => ("none" in cell ? undefined : `${engines[key].repo}/blob/${engines[key].tag}/${cell.path}#L${cell.line}`);

export const enginesIntro = [
  `Each chapter of this book builds one idea. vLLM and llama.cpp build the same ideas, in more code and for more models. This table says which file to open next: the file and the function or class, checked by reading the code at vLLM ${engines.vllm.tag} and llama.cpp ${engines.llamacpp.tag} on ${enginesChecked}. Links go to that exact version, so the line numbers hold.`,
  "vLLM is Python on top of PyTorch: most of its kernels come from CUDA extensions it builds, FlashAttention and cuBLAS, and the row points at vLLM's own file that makes the call. llama.cpp is C and C++ on ggml, which has its own CUDA kernels for nearly every operation.",
];

export const engineRows: EngineRow[] = [
  {
    slug: "one-token",
    idea: "text to token IDs, through the chat template",
    vllm: { path: "vllm/renderers/hf.py", line: 854, symbol: "safe_apply_chat_template", note: "The chat template turns the messages into one string here; the Hugging Face tokenizer then turns it into IDs (vllm/tokenizers/hf.py wraps it)." },
    llamacpp: { path: "src/llama-vocab.cpp", line: 4645, symbol: "llama_vocab::tokenize", note: "llama.cpp has its own tokenizer, read from the GGUF file. Chat templates are in src/llama-chat.cpp (built in) and common/chat.cpp (Jinja)." },
  },
  {
    slug: "foundations",
    idea: "timing a step and comparing it with the roofline",
    vllm: { path: "vllm/benchmarks/latency.py", line: 80, symbol: "main", note: "vllm bench latency times whole requests with time.perf_counter. No roofline in the code: the comparison with memory bandwidth is yours to make." },
    llamacpp: { path: "tools/llama-bench/llama-bench.cpp", line: 1477, symbol: "test", note: "llama-bench runs prompt and generation tests several times and keeps each run's nanoseconds in samples_ns. Again no roofline, only tokens per second." },
  },
  {
    slug: "gemm",
    idea: "matrix products, and the GEMV of a decode step",
    vllm: { path: "vllm/model_executor/layers/utils.py", line: 84, symbol: "default_unquantized_gemm", note: "BF16 weights go to torch.nn.functional.linear, which PyTorch hands to cuBLAS. vLLM writes no GEMM of its own for unquantized weights." },
    llamacpp: { path: "ggml/src/ggml-cuda/ggml-cuda.cu", line: 2446, symbol: "ggml_cuda_mul_mat", note: "Picks a kernel per call: a matrix-vector kernel (mmvf) for a few rows, its own quantized kernels (mmvq, mmq), and cuBLAS as the fallback." },
  },
  {
    slug: "rope",
    idea: "rotating queries and keys by position",
    vllm: { path: "vllm/model_executor/layers/rotary_embedding/base.py", line: 221, symbol: "RotaryEmbedding.forward_cuda", note: "Calls ops.rotary_embedding, a CUDA kernel in csrc/pos_encoding_kernels.cu that rotates query and key in place." },
    llamacpp: { path: "ggml/src/ggml-cuda/rope.cu", line: 714, symbol: "ggml_cuda_op_rope", note: "rope_norm pairs neighbouring dimensions, rope_neox pairs each with the one half the width away; the model decides which. The backend can also fuse RoPE with the write into the cache." },
  },
  {
    slug: "paged-kv",
    idea: "the KV cache and who owns which part of it",
    vllm: { path: "vllm/v1/core/block_pool.py", line: 107, symbol: "BlockPool", note: "get_new_blocks and free_blocks hand out fixed-size blocks. Each request's block table is in vllm/v1/worker/gpu/block_table.py." },
    llamacpp: { path: "src/llama-kv-cells.h", line: 68, symbol: "llama_kv_cells", note: "No pages: each layer has one K and one V tensor of kv_size cells. A cell records its position and the sequences using it; llama_kv_cache::find_slot looks for free cells." },
  },
  {
    slug: "paged-attention",
    idea: "decode attention over the cached keys and values",
    vllm: { path: "vllm/v1/attention/backends/flash_attn.py", line: 1066, symbol: "FlashAttentionImpl", note: "forward passes the block table to FlashAttention, which reads keys and values straight from the blocks. There is no separate decode kernel." },
    llamacpp: { path: "ggml/src/ggml-cuda/fattn.cu", line: 819, symbol: "ggml_cuda_flash_attn_ext", note: "Picks the vector kernel for one query row and a tensor core kernel for more. With flash attention off, attention is a matrix product, a softmax and another product." },
  },
  {
    slug: "attention-v2",
    idea: "online softmax: a running max and a running sum",
    vllm: { path: "vllm/v1/attention/ops/triton_attention_helpers.py", line: 462, symbol: "softmax_step", note: "Easy to read in Triton: the new max m_j, alpha rescales the old sum, L_new = L * alpha + l_j. FlashAttention does the same inside its CUDA code." },
    llamacpp: { path: "ggml/src/ggml-cuda/fattn-vec.cuh", line: 21, symbol: "flash_attn_ext_vec", note: "KQ_max and KQ_sum are the running max and sum; KQ_max_scale rescales what was added up before a new max." },
  },
  {
    slug: "split-attention",
    idea: "one long context split over many blocks, then merged",
    vllm: { path: "vllm/v1/attention/ops/triton_unified_attention.py", line: 814, symbol: "reduce_segments", note: "Long decodes run the 3D kernel: each segment writes its own max, sum and partial output, and reduce_segments merges them. FlashAttention's num_splits is the same idea." },
    llamacpp: { path: "ggml/src/ggml-cuda/fattn-common.cuh", line: 975, symbol: "flash_attn_combine_results", note: "parallel_blocks is how many blocks share one context; this kernel merges their partial outputs with each block's max and sum." },
  },
  {
    slug: "flash-prefill",
    idea: "attention for many query rows at once, in tiles",
    vllm: { path: "vllm/vllm_flash_attn/flash_attn_interface.py", line: 176, symbol: "flash_attn_varlen_func", note: "vLLM's own build of FlashAttention. Prompt rows and decode rows of one step go into the same varlen call." },
    llamacpp: { path: "ggml/src/ggml-cuda/fattn-mma-f16.cuh", line: 1911, symbol: "flash_attn_ext_f16", note: "The tensor core flash attention kernel, used when many query rows come at once, as in prompt processing." },
  },
  {
    slug: "add-rmsnorm",
    idea: "the residual add and RMSNorm",
    vllm: { path: "vllm/model_executor/layers/layernorm.py", line: 39, symbol: "RMSNorm", note: "Given a residual it calls fused_add_rms_norm, the add and the norm in one kernel (csrc/layernorm_kernels.cu), the same fusion as this engine." },
    llamacpp: { path: "ggml/src/ggml-cuda/norm.cu", line: 640, symbol: "ggml_cuda_op_rms_norm_fused", note: "The residual add stays its own ggml_add in the graph (src/models/llama.cpp); the CUDA backend fuses the norm with the weight multiply after it." },
  },
  {
    slug: "swiglu",
    idea: "the gated MLP activation",
    vllm: { path: "vllm/model_executor/layers/activation.py", line: 116, symbol: "SiluAndMul", note: "silu of the first half times the second half, one CUDA kernel (csrc/activation_kernels.cu). Gate and up come out of one merged matrix product." },
    llamacpp: { path: "ggml/src/ggml-cuda/unary.cu", line: 401, symbol: "ggml_cuda_op_swiglu", note: "The graph asks for it in build_ffn (src/llama-graph.cpp). For a few rows the backend can fuse the gate and up products and the GLU into one kernel." },
  },
  {
    slug: "sampling",
    idea: "turning logits into the next token",
    vllm: { path: "vllm/v1/worker/gpu/sample/sampler.py", line: 42, symbol: "Sampler", note: "Temperature, min-p, top-k and top-p, then a Gumbel sample (gumbel.py next to it), for the whole batch on the GPU." },
    llamacpp: { path: "common/sampling.cpp", line: 191, symbol: "common_sampler_init", note: "Builds a sampler chain from src/llama-sampler.cpp: top-k, top-p, temperature and the others, then dist draws the token." },
  },
  {
    slug: "cuda-graphs",
    idea: "replaying a step's kernel launches as one graph",
    vllm: { path: "vllm/v1/worker/gpu/cudagraph_utils.py", line: 154, symbol: "CudaGraphManager", note: "Graphs are captured for a list of batch sizes at startup (capture_model in model_runner.py), and a step is padded up to the nearest one." },
    llamacpp: { path: "ggml/src/ggml-cuda/ggml-cuda.cu", line: 5172, symbol: "ggml_cuda_graph_evaluate_and_capture", note: "Captures the ggml graph as a CUDA graph and replays it while ggml_cuda_graph_update_required finds nothing changed, the usual case one token at a time." },
  },
  {
    slug: "speculative-decoding",
    idea: "a cheap guess, checked in one pass",
    vllm: { path: "vllm/v1/worker/gpu/spec_decode/standalone_ar/speculator.py", line: 33, symbol: "StandaloneARSpeculator", note: "A separate small model drafts; RejectionSampler in spec_decode/rejection_sampler.py checks the draft. Prompt lookup is NgramGPUSpeculator in spec_decode/ngram." },
    llamacpp: { path: "common/speculative.cpp", line: 2916, symbol: "common_speculative_init", note: "Sets up the drafters from one list of types: a draft model (draft-simple), n-gram lookup (ngram-simple, ngram-cache) and others." },
  },
  {
    slug: "tensor-parallelism",
    idea: "one model cut across several GPUs",
    vllm: { path: "vllm/model_executor/layers/linear.py", line: 1686, symbol: "RowParallelLinear", note: "ColumnParallelLinear cuts a weight by output columns, RowParallelLinear by input rows and adds the parts with tensor_model_parallel_all_reduce." },
    llamacpp: { path: "src/llama-model.cpp", line: 415, symbol: "llama_meta_device_get_split_state", note: "The default --split-mode layer gives each GPU whole layers. --split-mode tensor, marked experimental, cuts weights and KV; this decides the cut per tensor." },
  },
  {
    slug: "memory-budget",
    idea: "how much memory the KV cache gets",
    vllm: { path: "vllm/v1/worker/gpu_worker.py", line: 673, symbol: "Worker.determine_available_memory", note: "A dummy forward pass measures the peak; the KV cache gets what is left under gpu_memory_utilization (0.92 by default, 1.0 with the extensible KV cache)." },
    llamacpp: { path: "common/fit.cpp", line: 980, symbol: "common_fit_params", note: "The cache is sized by the context length (-c) and allocated whole. With --fit on, the default, settings left unset, the context among them, are lowered until it fits." },
  },
  {
    slug: "serving",
    idea: "the scheduler and continuous batching",
    vllm: { path: "vllm/v1/core/sched/scheduler.py", line: 716, symbol: "Scheduler.schedule", note: "Every step: running requests first, then waiting ones while the token budget and free blocks last. When blocks run out it preempts a running request." },
    llamacpp: { path: "tools/server/server-context.cpp", line: 3558, symbol: "update_slots", note: "The server has a fixed number of slots (--parallel). Each step puts every generating slot's next token in one batch, then fills it with prompt tokens." },
  },
  {
    slug: "chunked-prefill",
    idea: "a long prompt cut into pieces that share steps with decode",
    vllm: { path: "vllm/config/scheduler.py", line: 146, symbol: "SchedulerConfig.enable_chunked_prefill", note: "Lets a prompt be cut to what is left of max_num_batched_tokens in a step; the rest waits for the next step. Scheduler.schedule makes the cut." },
    llamacpp: { path: "tools/server/server-context.cpp", line: 3989, symbol: "update_slots", note: "Prompt tokens join after the decode tokens, up to n_batch per step, so a long prompt takes several steps. llama_context::decode cuts each batch into n_ubatch pieces." },
  },
  {
    slug: "where-the-time-went",
    idea: "profiling a step kernel by kernel",
    vllm: { path: "vllm/profiler/wrapper.py", line: 682, symbol: "CudaProfilerWrapper", note: "Starts and stops the CUDA profiler around a window so Nsight Systems records only that part. TorchProfilerWrapper in the same file uses the PyTorch profiler." },
    llamacpp: { path: "src/llama-context.cpp", line: 5117, symbol: "llama_perf_context_print", note: "Prints prompt and generation time per token. There is no per-kernel breakdown in the code: Nsight Systems run from outside is the way to get one." },
  },
  {
    slug: "quantization",
    idea: "fewer bytes per weight, with a scale to undo it",
    vllm: { path: "vllm/model_executor/layers/quantization/fp8.py", line: 230, symbol: "Fp8LinearMethod", note: "Fp8Config reads the checkpoint's quantization settings. Each FP8 weight gets a per-tensor weight_scale, and init_fp8_linear_kernel picks the scaled matrix-product kernel. FP8 KV cache scales (k_scale, v_scale) are in kv_cache.py next to it." },
    llamacpp: { path: "ggml/src/ggml-cuda/mmvq.cu", line: 1452, symbol: "ggml_cuda_mul_mat_vec_q", note: "At decode the CUDA backend's matrix-vector product takes the quantized blocks as its input, such as block_q4_0: 32 weights and one scale d (ggml/src/ggml-common.h). The types, F8_E4M3 among them, are enum ggml_type in ggml.h; llama-quantize writes them (src/llama-quant.cpp)." },
  },
];

export const engineRow = (slug: string) => engineRows.find((r) => r.slug === slug);
