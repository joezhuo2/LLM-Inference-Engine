import hardwareJson from "@/data/hardware.json";

export type GlossEntry = { key: string; term: string; full: string; why: string; aliases: string[] };

const hw = hardwareJson as { sms: number; vram_gb: number; l2_kb: number; shared_kb_per_sm: number };

const entries: GlossEntry[] = [
  { key: "flop", term: "FLOP", full: "floating-point operation", why: "One multiply or one add on floating-point numbers. Dividing a kernel's FLOPs by the bytes it moves tells you whether math or memory limits it.", aliases: ["FLOPs", "FLOP"] },
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
  { key: "graph", term: "CUDA Graph", full: "a recorded chain of kernel launches", why: "Recorded once, then replayed with a single launch. The GPU gets the whole chain at once, so it no longer idles a few microseconds before each kernel.", aliases: ["CUDA Graphs", "CUDA Graph", "CUDA graphs", "CUDA graph"] },
  { key: "launch", term: "kernel launch", full: "the CPU asking the GPU to run one kernel", why: "Each kernel launched on its own leaves the GPU idle for a few microseconds before it starts, which matters when the kernel itself only runs for a few.", aliases: ["kernel launches", "kernel launch"] },
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
  { key: "prefill", term: "prefill", full: "processing the prompt and filling the KV cache", why: "The prompt's tokens go through the model together, in one step or in chunks over several. With many tokens per step these are large matrix multiplies, usually limited by math.", aliases: ["prefill"] },
  { key: "decode", term: "decode", full: "generating one token per step", why: "After prefill, each step adds one token. It reads every weight for very little math, so memory bandwidth limits it.", aliases: ["decode"] },
  { key: "kernel", term: "kernel", full: "a function that runs on the GPU", why: "Written once and run by thousands of threads at the same time, each on its own piece of the data. A decode step is a chain of a few hundred of them.", aliases: ["kernels", "kernel"] },
  { key: "thread", term: "thread", full: "one instance of a running kernel", why: "Each thread runs the same code on its own index, so it knows which values are its job. Threads are grouped into warps of 32 and blocks of up to 1,024.", aliases: ["threads", "thread"] },
  { key: "grid", term: "grid", full: "all the blocks of one kernel launch", why: "A launch asks for a grid of blocks; the GPU spreads them over the SMs. A grid smaller than the GPU leaves SMs with nothing to do.", aliases: ["grids", "grid"] },
  { key: "wave", term: "wave", full: "the blocks that fit on the GPU at once", why: `The GPU holds a fixed number of blocks at once: blocks per SM times its ${hw.sms} SMs. A grid up to that size runs as one wave; a bigger one needs a second wave, which may be only partly full and still costs nearly as long as a full one.`, aliases: ["waves", "wave"] },
  { key: "lane", term: "lane", full: "one thread's position in its warp, 0 to 31", why: "Lanes of a warp run the same instruction together and can read each other's registers with shuffles.", aliases: ["lanes", "lane"] },
  { key: "shuffle", term: "warp shuffle", full: "a register exchange between lanes of a warp", why: "One instruction hands a value from one lane to another without going through shared memory. An xor shuffle pairs each lane with the lane whose index differs in one bit, so five of them sum a whole warp.", aliases: ["xor shuffles", "warp shuffles", "shuffles", "shuffle"] },
  { key: "barrier", term: "barrier", full: "a point every thread of a block must reach", why: "__syncthreads() makes the block wait until all its threads arrive, so data written to shared memory is ready to read. Each one costs time while the slowest warp catches up.", aliases: ["barriers", "barrier", "__syncthreads"] },
  { key: "reduction", term: "reduction", full: "combining many values into one", why: "A sum or a maximum over a row. On a GPU it runs as a tree: pairs combine, then pairs of pairs, so 1,024 values take 10 rounds instead of 1,023 steps.", aliases: ["reductions", "reduction"] },
  { key: "coalesced", term: "coalesced loads", full: "a warp's loads merged into few memory transactions", why: "When the 32 lanes of a warp read neighbouring addresses, the hardware fetches them as a few whole sectors. Scattered addresses each cost their own transaction.", aliases: ["coalesced", "coalesce", "coalesces"] },
  { key: "sector", term: "sector", full: "32 bytes, the smallest piece DRAM delivers", why: "Memory is read in whole sectors, so a kernel that needs 2 bytes from a sector still pays for 32.", aliases: ["sectors", "sector"] },
  { key: "vecload", term: "16-byte loads", full: "one instruction that moves 8 BF16 values", why: "A vectorized load (128 bits) moves 8 BF16 values in one instruction instead of 8, so threads spend their time waiting on memory, not issuing loads.", aliases: ["16-byte loads", "16-byte load", "128-bit loads", "128-bit load", "128-bit stores"] },
  { key: "latency", term: "latency", full: "how long one operation takes to finish", why: "In serving, how long a user waits for a token. Inside the GPU, how long one load takes: a DRAM load waits hundreds of cycles, hidden only by switching to another warp, so a kernel with too few warps sits idle even when memory is not busy.", aliases: ["latency"] },
  { key: "bandwidth", term: "bandwidth", full: "bytes moved per second", why: "How fast memory can deliver data. Decode reads every weight per token, so DRAM bandwidth sets its speed limit.", aliases: ["bandwidth"] },
  { key: "membound", term: "memory bound", full: "limited by bytes moved, not by math", why: "The kernel would finish sooner with faster memory, not faster arithmetic. It sits left of the ridge on the roofline.", aliases: ["memory-bound", "memory bound"] },
  { key: "computebound", term: "compute bound", full: "limited by math, not by bytes moved", why: "The kernel would finish sooner with more FLOP/s. It sits right of the ridge, where each byte read is used many times.", aliases: ["compute-bound", "compute bound"] },
  { key: "ridge", term: "ridge", full: "where the roofline's slope meets its flat top", why: "Peak FLOP/s divided by bandwidth: the FLOPs per byte an operation needs before math, not memory, limits it.", aliases: ["ridge point", "ridge"] },
  { key: "dependent", term: "dependent loads", full: "loads that wait on an earlier result", why: "A thread that needs one value to know what to read next cannot overlap the two waits, so their latencies add up.", aliases: ["dependent loads", "dependent load"] },
  { key: "rowmajor", term: "row-major", full: "a matrix stored one row after another", why: "C and PyTorch store matrices row by row; Fortran and cuBLAS read them column by column. The same bytes read the other way are the transpose.", aliases: ["row-major", "column-major"] },
  { key: "ld", term: "leading dimension", full: "the stride between rows or columns in memory", why: "Tells cuBLAS how many elements to skip to reach the next column, so it can walk a matrix stored in any layout.", aliases: ["leading dimension"] },
  { key: "tiling", term: "tiling", full: "cutting a matrix into blocks that fit on chip", why: "A block loads a tile of A and a tile of B into shared memory once and reuses every value many times, so a big GEMM reads each byte from DRAM only a few times.", aliases: ["tiling", "tiles", "tile"] },
  { key: "epilogue", term: "epilogue", full: "work a GEMM does on its result before writing it", why: "An activation or an add folded into the matrix multiply's last step, so the result never makes an extra trip to memory.", aliases: ["epilogue"] },
  { key: "eager", term: "eager mode", full: "launching every kernel one by one from the CPU", why: "The opposite of replaying a CUDA Graph: each kernel pays its own launch and the gap before it.", aliases: ["eager mode", "eagerly"] },
  { key: "capture", term: "stream capture", full: "recording launches into a graph instead of running them", why: "Between begin and end capture, kernels on that stream are added to a CUDA Graph, to be replayed later with one launch.", aliases: ["stream capture"] },
  { key: "fp64", term: "FP64", full: "64-bit floating point", why: "Double precision. Laptop GPUs run it at a small fraction of their FP32 rate, so one double-precision log per token can dominate a kernel.", aliases: ["FP64"] },
  { key: "philox", term: "Philox", full: "a counter-based random number generator", why: "Gives the random number for any (seed, counter) directly, with no state to carry, so a token's noise is the same alone or in a batch.", aliases: ["Philox"] },
  { key: "argmax", term: "argmax", full: "the index of the largest value", why: "Greedy decoding is the argmax of the logits: the single most likely next token.", aliases: ["argmax"] },
  { key: "online", term: "online softmax", full: "softmax computed in one pass over chunks", why: "Keeps a running maximum and sum and rescales them when a larger score arrives, so attention never stores every score at once.", aliases: ["online softmax", "online-softmax"] },
  { key: "flash", term: "FlashAttention", full: "attention computed block by block on chip", why: "Uses an online softmax over blocks of keys, so a prompt's whole table of scores never goes to DRAM, and each block of keys and values is loaded once for a block of query rows instead of once per row.", aliases: ["FlashAttention"] },
  { key: "head", term: "attention head", full: "one independent slice of attention", why: "Each query head has its own 64-value slice of the query and looks for its own pattern. Its keys and values come from a key and value head, which several query heads share in TinyLlama. The heads' outputs are joined afterwards.", aliases: ["attention heads", "attention head", "query heads", "query head", "key heads", "KV heads", "KV head"] },
  { key: "gqa", term: "GQA", full: "grouped-query attention", why: "Several query heads share one key head and one value head (8 to 1 in TinyLlama), which shrinks the KV cache by the same factor.", aliases: ["grouped-query attention", "GQA"] },
  { key: "blocktable", term: "block table", full: "a request's list of KV cache blocks", why: "Maps a request's token positions to the 16-token blocks that hold their keys and values, wherever they sit in the pool.", aliases: ["block tables", "block table"] },
  { key: "residual", term: "residual stream", full: "the running vector each layer adds to", why: "Every layer reads it, computes something and adds the result back, so information from earlier layers is never overwritten.", aliases: ["residual stream"] },
  { key: "goodput", term: "goodput", full: "requests per second that meet their latency targets", why: "Throughput that counts only requests whose time to first token and time per token both stay under a limit.", aliases: ["goodput"] },
  { key: "preempt", term: "preemption", full: "pausing a running request to free its memory", why: "When the KV pool runs out, the scheduler takes a request's blocks back and queues it to start again later.", aliases: ["preemption", "preempted", "preempts"] },
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
