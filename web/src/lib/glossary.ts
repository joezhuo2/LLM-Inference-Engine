import hardwareJson from "@/data/hardware.json";

export type GlossEntry = { key: string; term: string; full: string; why: string; aliases: string[] };

const hw = hardwareJson as { sms: number; vram_gb: number; l2_kb: number; shared_kb_per_sm: number };

const entries: GlossEntry[] = [
  { key: "flop", term: "FLOP", full: "floating-point operation", why: "One multiply or one add on decimal numbers. Dividing a kernel's FLOPs by the bytes it moves tells you whether math or memory limits it.", aliases: ["FLOPs", "FLOP"] },
  { key: "tflops", term: "TFLOP/s", full: "trillion floating-point operations per second", why: "A rate of math. The GPU's measured peak is the flat ceiling on the roofline that a compute-bound kernel runs into.", aliases: ["TFLOP/s", "FLOP/s"] },
  { key: "sm", term: "SM", full: "streaming multiprocessor", why: `One of the GPU's independent processors, each with its own registers and shared memory. This GPU has ${hw.sms}; a kernel that keeps only a few busy leaves the rest idle.`, aliases: ["SMs", "SM"] },
  { key: "dram", term: "DRAM", full: "dynamic random-access memory", why: `The GPU's main memory, off the chip, holding the weights and the KV cache (${hw.vram_gb.toFixed(1)} GB here). It is the slowest tier, and most decode kernels are limited by how fast they read it.`, aliases: ["DRAM"] },
  { key: "vram", term: "VRAM", full: "video random-access memory", why: "The GPU's own memory, the same DRAM: it must fit the model's weights, the KV cache and every scratch buffer.", aliases: ["VRAM"] },
  { key: "l2", term: "L2", full: "level-2 cache", why: `A ${hw.l2_kb / 1024} MB cache on the chip shared by every SM. Data read again soon after can come from here instead of DRAM, several times faster.`, aliases: ["L2"] },
  { key: "smem", term: "shared memory", full: "on-chip scratchpad, one per SM", why: `A small fast memory (${hw.shared_kb_per_sm} KB per SM) that the threads of one block share and manage by hand, so data loaded once from DRAM can be reused many times.`, aliases: ["shared memory", "smem"] },
  { key: "registers", term: "registers", full: "per-thread storage inside the SM", why: "The fastest memory there is, private to each thread. A kernel that needs many per thread leaves room for fewer threads on each SM.", aliases: ["registers", "register"] },
  { key: "warp", term: "warp", full: "a group of 32 threads", why: "The SM runs a warp's 32 threads in lockstep, one instruction for all of them, and they can pass values to each other directly with shuffles.", aliases: ["warps", "warp"] },
  { key: "block", term: "thread block", full: "a group of up to 1,024 threads on one SM", why: "Threads in a block can share memory and wait for each other. A kernel launch is a grid of blocks spread over the SMs.", aliases: ["thread blocks", "thread block"] },
  { key: "occupancy", term: "occupancy", full: "resident warps compared with the SM's maximum", why: "More warps on an SM let it hide memory latency by switching to another warp while one waits for data.", aliases: ["occupancy"] },
  { key: "tensor", term: "tensor cores", full: "matrix-multiply units inside each SM", why: "They multiply small matrix tiles in one instruction, far faster than ordinary cores; cuBLAS uses them for the big GEMMs.", aliases: ["tensor cores", "tensor core"] },
  { key: "cuda", term: "CUDA", full: "Compute Unified Device Architecture", why: "NVIDIA's platform and C++ dialect for code that runs on the GPU. Every kernel in this engine is a CUDA kernel.", aliases: ["CUDA"] },
  { key: "graph", term: "CUDA Graph", full: "a recorded chain of kernel launches", why: "Recorded once, then replayed with a single launch, which removes the CPU's cost of launching each kernel on its own.", aliases: ["CUDA Graphs", "CUDA Graph", "CUDA graphs", "CUDA graph"] },
  { key: "launch", term: "kernel launch", full: "the CPU asking the GPU to run one kernel", why: "Each launch costs a few microseconds of overhead, which matters when the kernel itself only runs for a few.", aliases: ["kernel launches", "kernel launch"] },
  { key: "gemm", term: "GEMM", full: "general matrix multiply", why: "Multiplying two matrices, C = A·B. Every linear layer in the model is one; with a single token it shrinks to a GEMV.", aliases: ["GEMMs", "GEMM"] },
  { key: "gemv", term: "GEMV", full: "general matrix-vector multiply", why: "A matrix times one vector, which is what each linear layer does for a single token. Every weight is used once, so DRAM bandwidth sets the speed, not math.", aliases: ["GEMVs", "GEMV"] },
  { key: "cublas", term: "cuBLAS", full: "CUDA Basic Linear Algebra Subprograms", why: "NVIDIA's library of tuned matrix routines. The engine calls it for the matrix multiplies instead of writing its own.", aliases: ["cuBLAS"] },
  { key: "kv", term: "KV cache", full: "key-value cache", why: "The keys and values attention made for every earlier token, kept in GPU memory so each new token only computes its own instead of redoing the whole prompt.", aliases: ["KV cache", "KV"] },
  { key: "qkv", term: "QKV", full: "query, key and value", why: "The three vectors attention makes from each token: the query is matched against every key, and the match decides how the values are mixed.", aliases: ["QKV"] },
  { key: "softmax", term: "softmax", full: "scores turned into probabilities", why: "Exponentiate each score and divide by the sum, so they are all positive and add up to 1. Used on attention scores and on the logits.", aliases: ["softmax"] },
  { key: "logits", term: "logits", full: "raw scores for every vocabulary token", why: "The model's last output: one number for each of the 32,000 tokens it knows, before softmax turns them into probabilities.", aliases: ["logits", "logit"] },
  { key: "lmhead", term: "LM head", full: "language-model head", why: "The last matrix multiply, which turns the final hidden vector into one logit per vocabulary token.", aliases: ["LM head"] },
  { key: "mlp", term: "MLP", full: "multilayer perceptron", why: "The feed-forward half of each layer: widen the vector, apply SwiGLU, project it back down. Most of the model's weights live here.", aliases: ["MLP"] },
  { key: "rmsnorm", term: "RMSNorm", full: "root-mean-square normalization", why: "Scales a vector so its root mean square is 1, then by learned weights. It runs before attention and before the MLP in every layer.", aliases: ["RMSNorm"] },
  { key: "swiglu", term: "SwiGLU", full: "Swish-gated linear unit", why: "The MLP's activation: one projection goes through SiLU and is multiplied, element by element, by a second projection.", aliases: ["SwiGLU"] },
  { key: "rope", term: "RoPE", full: "rotary position embedding", why: "Tells attention where each token sits by rotating pairs of query and key dimensions by an angle that grows with position.", aliases: ["RoPE"] },
  { key: "bf16", term: "BF16", full: "bfloat16 (brain floating point)", why: "A 16-bit number with FP32's range but only 8 bits of precision. The model is stored in it, which halves the bytes every kernel has to move.", aliases: ["BF16"] },
  { key: "fp32", term: "FP32", full: "32-bit floating point", why: "Ordinary single precision. Kernels often load BF16 but add up in FP32 so rounding errors don't pile up.", aliases: ["FP32"] },
  { key: "fma", term: "FMA", full: "fused multiply-add", why: "a × b + c in one instruction with a single rounding. Matching PyTorch bit for bit sometimes means forbidding it.", aliases: ["FMA"] },
  { key: "ulp", term: "ulp", full: "unit in the last place", why: "The gap between a number and the next one the format can hold. Off by 1 ulp is the smallest possible rounding difference.", aliases: ["ulps", "ulp"] },
  { key: "prefill", term: "prefill", full: "processing the whole prompt in one pass", why: "Every prompt token goes through the model at once, filling the KV cache. Many tokens per step make it large matrix multiplies, limited by math.", aliases: ["prefill"] },
  { key: "decode", term: "decode", full: "generating one token per step", why: "After prefill, each step adds one token. It reads every weight for very little math, so memory bandwidth limits it.", aliases: ["decode"] },
  { key: "ttft", term: "TTFT", full: "time to first token", why: "From sending a request to getting the first generated token back: mostly waiting in the queue plus prefill.", aliases: ["TTFT"] },
  { key: "tpot", term: "TPOT", full: "time per output token", why: "The average gap between generated tokens after the first, which decides how fast streamed text feels.", aliases: ["TPOT"] },
  { key: "pct", term: "p50, p99", full: "50th and 99th percentile", why: "p50 is the median. p99 is the value only 1 request in 100 is slower than, a measure of the worst cases.", aliases: ["p50", "p95", "p99"] },
  { key: "toks", term: "tok/s", full: "tokens per second", why: "Throughput: how many tokens the engine produces each second, summed over every request it is serving.", aliases: ["tok/s"] },
  { key: "roofline", term: "roofline", full: "performance against FLOPs per byte", why: "A kernel sits under a sloped memory roof or a flat compute roof. Where it sits shows what limits it and how far it is from that limit.", aliases: ["roofline"] },
  { key: "ncu", term: "ncu", full: "NVIDIA Nsight Compute", why: "A profiler that replays one kernel and reports its time, bytes moved and hardware counters. Most kernel numbers on this site come from it.", aliases: ["ncu"] },
  { key: "nsys", term: "nsys", full: "NVIDIA Nsight Systems", why: "A profiler that records a timeline of the whole program: every kernel, copy and CPU call, and the gaps between them.", aliases: ["nsys"] },
  { key: "sanitizer", term: "compute-sanitizer", full: "NVIDIA's GPU memory checker", why: "Runs a program and reports out-of-bounds accesses and leaked memory on the GPU. memcheck is its default mode.", aliases: ["compute-sanitizer", "memcheck"] },
  { key: "gumbel", term: "Gumbel", full: "the Gumbel-max trick", why: "Add random Gumbel noise to every logit and take the largest: the same odds as sampling from the softmax, without computing it.", aliases: ["Gumbel"] },
  { key: "topk", term: "top-k", full: "keep the k most likely tokens", why: "Before sampling, every token outside the k highest logits is dropped, so unlikely tokens can never be picked.", aliases: ["top-k"] },
  { key: "topp", term: "top-p", full: "nucleus sampling", why: "Before sampling, keep the smallest set of tokens whose probabilities add up to p and drop the rest.", aliases: ["top-p"] },
  { key: "vllm", term: "vLLM", full: "an open-source LLM serving engine", why: "The project that introduced paged attention; used here as a baseline to compare against.", aliases: ["vLLM"] },
  { key: "llamacpp", term: "llama.cpp", full: "an open-source C/C++ inference engine", why: "A popular engine for running models on local hardware; the other baseline this engine is measured against.", aliases: ["llama.cpp"] },
];

export const glossary: Record<string, GlossEntry> = Object.fromEntries(entries.map((e) => [e.key, e]));

const byAlias = new Map<string, string>();
for (const e of entries) for (const a of e.aliases) byAlias.set(a, e.key);

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const pattern = new RegExp(`(?<![\\w/.-])(${[...byAlias.keys()].sort((a, b) => b.length - a.length).map(escape).join("|")})(?![\\w/-])`, "g");

export type TermMatch = { key: string; start: number; end: number };

export function findTerms(text: string): TermMatch[] {
  const out: TermMatch[] = [];
  for (const m of text.matchAll(pattern)) {
    const key = byAlias.get(m[1]);
    if (key && m.index !== undefined) out.push({ key, start: m.index, end: m.index + m[1].length });
  }
  return out;
}

export type GlossPiece = { text: string; term?: string };

export function splitTerms(text: string, seen?: Set<string>): GlossPiece[] {
  const pieces: GlossPiece[] = [];
  let at = 0;
  for (const m of findTerms(text)) {
    if (seen?.has(m.key)) continue;
    seen?.add(m.key);
    if (m.start > at) pieces.push({ text: text.slice(at, m.start) });
    pieces.push({ text: text.slice(m.start, m.end), term: m.key });
    at = m.end;
  }
  if (at < text.length) pieces.push({ text: text.slice(at) });
  return pieces;
}
