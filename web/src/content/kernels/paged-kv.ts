import { kvDemo, kvReplay } from "@/lib/data";
import type { KernelStory } from "./types";

const replay = kvReplay();
const demo = kvDemo();

export const pagedKv: KernelStory = {
  slug: "paged-kv",
  index: 7,
  title: "paged KV cache",
  short: "write_kv and the block manager: one pool of 16-token blocks on the GPU, handed out as requests grow and taken back when they finish.",
  formula: "slot = table[pos / 16] · 16 + pos mod 16",
  headline: {
    value: `${replay.peak.blocks} / ${replay.meta.kv_blocks.toLocaleString("en-US")}`,
    label: `KV blocks held at the busiest step of a recorded ${replay.meta.num_requests}-request chat run, ${replay.peak.running} requests at once`,
  },
  lead: {
    title: "allocate it yourself",
    intro: `Three real chat requests, answered greedily by the engine on the ${demo.gpu.replace("NVIDIA GeForce ", "")} and recorded in results/kv_demo.json, served from a 64-block pool by the rules of the BlockManager and the scheduler: a request is admitted when blocks for its whole length are free, every decode step appends one slot, a full pool preempts the newest request. Add requests, then switch to contiguous to run the same arrivals with every request reserving its prompt plus the ${demo.max_new_tokens} tokens it may generate. Hover a block for its tokens and where it sits in memory.`,
  },
  sources: [
    { path: "src/kernels/write_kv.cu", label: "write_kv" },
    { path: "src/kv/block_manager.cpp", label: "BlockManager" },
    { path: "src/kv/cache_size.cpp", label: "block size and pool size" },
    { path: "src/runtime/kv_cache.cpp", label: "the pool on the GPU" },
    {
      path: "src/sched/paged_batch.cpp",
      label: "slots and block tables per step",
    },
    { path: "src/sched/scheduler.cpp", label: "admit, append, preempt, free" },
  ],
  tests: [
    {
      path: "tests/gpu/test_write_kv.cpp",
      label: "write_kv against a host reference",
    },
    { path: "tests/unit/test_block_manager.cpp", label: "BlockManager" },
    {
      path: "tests/unit/test_scheduler.cpp",
      label: "scheduler on a fake KV cache",
    },
    {
      path: "tests/gpu/test_serve.cpp",
      label: "16 concurrent against 16 solo, preemption",
    },
  ],
  testing:
    "The 8 write_kv tests fill both caches and guard bytes around them with random values, run the kernel and compare every byte with a host reference: a 37-token sequence through a shuffled block table, padding rows that must write nothing, the slots at block boundaries and at both ends of the cache, block sizes 1, 3 and 5 with odd head layouts, a 2,048-token prefill over 128 random blocks, a 16-sequence decode batch, and 11 rejected arguments. They were checked against nine deliberately broken versions before Joe's kernel existed. The 15 BlockManager tests cover block counts at lengths 1 to 2,048, the null block never handed out, a new block only at a multiple of 16, rejected calls changing nothing, and 10,000 random operations that keep every invariant. The 21 scheduler tests run on a fake KV cache filled exactly where write_kv would write. On the real model, 16 concurrent greedy requests give the tokens of 16 solo runs, and a 56-block cache that forces 6 preemptions gives the same outputs as a roomy one. The page's replay of the recorded trace is a check of its own: the build fails unless the block manager's model leaves exactly the trace's free block count after each of its steps.",
  excerpts: [
    {
      file: "src/kernels/write_kv.cu",
      from: "    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
      to: "    v_cache[cache_idx] = v_val;",
      caption: "write_kv: one thread per element of a token's K and V; the slot picks the block and the row inside it, a slot of -1 writes nothing",
    },
    {
      file: "src/kv/block_manager.cpp",
      from: "int BlockManager::append_slot(SeqId seq) {",
      to: "    return seq_data.table.back() * block_size_ + pos % block_size_;",
      caption: "append_slot: a new block only when the last one is full, then the slot of the new token",
    },
    {
      file: "src/kv/block_manager.cpp",
      from: "int BlockManager::allocate_block() {",
      to: '    throw std::runtime_error("[BlockManager] Out of free blocks despite prior check.");',
      caption: "allocate_block: the lowest free block, never block 0, which stays reserved as the null block",
    },
  ],
  measurements: [
    {
      doc: "results",
      section: "gpu-time-per-op-in-a-decode-step",
      table: 0,
      caption: "GPU time per op in a batch-1 decode step (write_kv is one launch per layer), from Nsight Systems on 4 requests of 64 prompt and 64 output tokens",
      command: "scripts/profile.sh nsys decode_step --max-batch 1 --num-requests 4 --prompt-len 64 --output-len 64",
    },
    {
      doc: "results",
      section: "throughput-and-latency",
      table: 0,
      caption: "what serving many requests from one pool buys: output tokens per second against the max batch, 32 requests of 64 prompt and 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --num-requests 32 --prompt-len 64 --output-len 64 --max-batch 16 (and 1, 2, 4, 8; scripts/reproduce.sh)",
    },
  ],
  timeline: [
    {
      date: "Oct 10",
      prs: [24],
      branch: "kv/block-manager",
      version: "stub",
      author: "claude",
      title: "The block manager's spec",
      text: "The bookkeeping half of the paged cache as a stub with its spec: block 0 reserved as a null block, a sequence of length L owning ceil(L / 16) blocks, position pos living at table[pos / 16] · 16 + pos mod 16, and every broken precondition throwing without changing anything. 15 tests came with it for Joe to make pass.",
    },
    {
      date: "Oct 10",
      prs: [25],
      branch: "kv/paged-cache",
      version: "stub",
      author: "claude",
      title: "One allocation for the whole pool",
      text: "KvCache takes all its blocks in one allocation at startup, laid out [block, kv head, 16, 64] per layer, so one head's 16 keys are a contiguous 2 KB tile. A block across 22 layers, keys and values, is 360,448 bytes, and the pool gets 90% of the GPU memory left after the weights. Debug builds fill it with NaN so a read of a slot nothing wrote shows up. The write_kv stub came with 8 tests, checked against nine broken versions of a throwaway kernel.",
    },
    {
      date: "Oct 11",
      prs: [29],
      branch: "kv/block-manager-impl",
      version: "impl",
      author: "both",
      title: "BlockManager",
      text: "Joe's block manager: a free flag per block, a count of free ones, and per sequence its length and block table. allocate_block takes the lowest free block after block 0. Fixes drafted with Claude: slot and append_slot returned block ids instead of slots, a leftover outer loop in allocate, and preconditions that threw the wrong exception type. All 15 tests pass.",
    },
    {
      date: "Oct 12",
      prs: [31],
      branch: "kernel/write-kv-impl",
      version: "impl",
      author: "both",
      title: "write_kv",
      text: "Joe's kernel: one thread per (token, KV head, dimension) element reads K and V from the end of the fused QKV row and writes them bit for bit into the slot's block, with every index 64-bit and a negative slot returning before any write. A test caught two spec slips, fixed with Claude: a QKV row with no query columns was accepted, and an empty batch returned before the argument checks ran.",
    },
    {
      date: "Oct 14",
      prs: [28, 33],
      branch: "sched/scheduler-impl",
      version: "impl",
      author: "joe",
      title: "Requests come and go",
      text: "The scheduler (stub and 21 tests drafted by Claude, implementation Joe's) is what drives the block manager: a request is admitted when blocks for its whole length are free, each decode step appends one slot per running sequence, a finished request frees its blocks, and when the pool is full the newest request is preempted, its blocks freed and its tokens recomputed later.",
    },
    {
      date: "Oct 15",
      prs: [34],
      branch: "model/paged-runner",
      version: "wiring",
      author: "claude",
      title: "16 at once, the same as 16 alone",
      text: "The batch builder turns each step into slots and padded block tables for write_kv and the attention kernel. 16 concurrent greedy requests produce exactly the tokens of 16 solo runs, and a 56-block cache, small enough to force 6 preemptions, gives the same outputs as a roomy one.",
      metric: {
        kind: "value",
        value: "16 / 16",
        label: "concurrent requests identical to their solo runs",
      },
    },
    {
      date: "Oct 25",
      prs: [58, 59],
      branch: "bench/chat-prompts",
      version: "trace",
      author: "claude",
      title: "A recorded run",
      text: `engine bench --trace writes every step of a run with the free block count after it. The trace on this page is ${replay.meta.num_requests} chat requests arriving at ${replay.meta.request_rate} per second: at its busiest step ${replay.peak.running} requests held ${replay.peak.blocks} of the ${replay.meta.kv_blocks.toLocaleString("en-US")} blocks.`,
      metric: {
        kind: "value",
        value: `${replay.peak.blocks}`,
        label: `blocks at the peak, of ${replay.meta.kv_blocks.toLocaleString("en-US")}`,
      },
    },
  ],
};
