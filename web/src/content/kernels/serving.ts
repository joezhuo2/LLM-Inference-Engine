import { kvReplay, loadRun, loadSlo, loadTrace, modelShape, profilingTable } from "@/lib/data";
import { type KernelStory, measured } from "./types";

const shape = modelShape();
const replay = kvReplay();
const load = "under-load";
const sweep = "batch-size-past-16";

export const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export const blockTokens = replay.meta.block_size;
export const headDim = shape.hidden / shape.heads;
export const kvValues = shape.kv_heads * headDim;
export const tokenBytes = 2 * shape.layers * kvValues * 2;
export const blockBytes = tokenBytes * blockTokens;
export const poolBlocks = replay.meta.kv_blocks;
export const usableBlocks = poolBlocks - 1;
export const poolGb = (poolBlocks * blockBytes) / 1e9;
export const poolTokens = usableBlocks * blockTokens;
export const fullContext = shape.max_positions;
export const fullBlocks = fullContext / blockTokens;
export const fullMb = (fullBlocks * blockBytes) / 1e6;
export const maxBatch = Number(replay.meta.max_batch);
export const reservedFit = Math.floor(usableBlocks / fullBlocks);
export const worstCaseBlocks = maxBatch * fullBlocks;
export const worstCasePct = (100 * worstCaseBlocks) / poolBlocks;
export const poolFraction = 90;
export const blocksFor = (tokens: number) => Math.ceil(tokens / blockTokens);

export const chatLengths = replay.requests.map((q) => q.final_length);
export const chatMean = chatLengths.reduce((n, l) => n + l, 0) / chatLengths.length;
export const heldBy = (requests: number) => Array.from({ length: requests }, (_, i) => blocksFor(chatLengths[i % chatLengths.length])).reduce((n, b) => n + b, 0);
export const chatFit = (() => {
  let n = 0;
  for (let held = 0; held + blocksFor(chatLengths[n % chatLengths.length]) <= usableBlocks; n++) held += blocksFor(chatLengths[n % chatLengths.length]);
  return n;
})();

export const engine = "engine + graphs";
export const budgetSystem = "engine + graphs, budget 256";
export const vllm = "vLLM";
export const loadRates = [...new Set(profilingTable(load, 0, "results").rows.map((r) => r[1]))];
export const loadCell = (system: string, qps: string, column: string) => measured(load, { system, QPS: qps }, column, 0, "results");
export const goodput = (system: string, qps: string) => loadCell(system, qps, "goodput req/s");
export const tpot = (system: string, qps: string) => loadCell(system, qps, "TPOT p50 ms");
export const ttftTail = (system: string, qps: string) => loadCell(system, qps, "TTFT p99 ms");
export const sloMet = (system: string, qps: string) => loadCell(system, qps, "SLO met");
const loadTable = profilingTable(load, 0, "results");
const preemptionsOf = (test: (system: string) => boolean) => loadTable.rows.filter((r) => test(r[0])).reduce((n, r) => n + Number(r[loadTable.header.indexOf("preemptions")]), 0);
export const loadPreemptions = preemptionsOf((s) => s.startsWith("engine"));
export const vllmPreemptions = preemptionsOf((s) => s === vllm);
export const engineSystems = [...new Set(loadTable.rows.filter((r) => r[0].startsWith("engine")).map((r) => r[0]))];
export const loadConfigs = engineSystems.length;

export function peak(system: string): { qps: string; value: number } {
  return loadRates.map((qps) => ({ qps, value: goodput(system, qps) })).reduce((a, b) => (b.value > a.value ? b : a));
}

export const engineBest = engineSystems.map((system) => ({ system, ...peak(system) })).reduce((a, b) => (b.value > a.value ? b : a));
export const loadPredatesFlash = loadRun().date.slice(0, 10) < "2026-11-10";
export const queueBound = (system: string, qps: string) => ttftTail(system, qps) > loadSlo().ttft_ms && tpot(system, qps) < loadSlo().tpot_ms;

export const lowRate = loadRates[0];
export const highRate = loadRates[loadRates.length - 1];
export const slo = loadSlo();
export const trace = loadTrace();

export type SweepRow = { batch: number; tpot: number; tokens: number; peak: number };
export const sweepRows: SweepRow[] = profilingTable(sweep, 0, "results").rows.map((r) => {
  const cell = (column: string) => measured(sweep, { "max batch": r[0] }, column, 0, "results");
  return { batch: Number(r[0]), tpot: cell("graphs TPOT p50 (ms)"), tokens: cell("graphs tokens/s"), peak: cell("peak KV blocks") };
});
export const sweepLast = sweepRows[sweepRows.length - 1];
export const firstOverPool = sweepRows.find((r) => r.batch * fullBlocks > usableBlocks);

const run = loadRun();
const best = { engine: peak(engine), budget: peak(budgetSystem), vllm: peak(vllm) };

export const serving: KernelStory = {
  slug: "serving",
  index: 13,
  book: true,
  about: true,
  title: "how many requests fit",
  short: `The scheduler's three decisions: who is admitted, who is evicted when the KV pool is full, and how many rows a step may hold. With what it bought under load${peak(vllm).value > engineBest.value ? ", and why vLLM still won" : ""}.`,
  formula: `Σ ceil(length / ${blockTokens}) ≤ blocks − 1, running ≤ max_batch`,
  headline: {
    value: `${fmt(worstCaseBlocks)} / ${fmt(poolBlocks)}`,
    label: `KV blocks that ${maxBatch} requests hold if every one reaches the full ${fmt(fullContext)}-token context, of the pool${loadPreemptions === 0 ? ": max_batch, not memory, was the limit in every load run" : ""}`,
  },
  sandbox: {
    title: "try it yourself",
    intro: `Two workbenches. The first serves three real chat requests from a small block pool by the rules of the BlockManager and the scheduler: a request is admitted when blocks for its whole length are free, every decode step appends one slot, a full pool preempts the newest request. It has no max_batch, so the pool is its only limit. Add requests, then switch to contiguous to run the same arrivals with every request reserving its prompt plus the tokens it may generate. The second replays a run the engine recorded, step by step: which requests shared each step and which blocks they held.`,
  },
  sources: [
    { path: "src/sched/scheduler.cpp", label: "admit, decode, preempt, retire" },
    { path: "include/engine/sched/scheduler.h", label: "SchedulerConfig, Step" },
    { path: "src/kv/block_manager.cpp", label: "can_allocate, can_append, free" },
    { path: "src/kv/cache_size.cpp", label: "bytes per block, blocks in the pool" },
    { path: "scripts/load_sweep.sh", label: "the load sweep" },
    { path: "docs/baselines.md", label: "how vLLM was run" },
  ],
  tests: [
    { path: "tests/unit/test_scheduler.cpp", label: "21 scheduler tests on a fake KV cache" },
    { path: "tests/unit/test_block_manager.cpp", label: "BlockManager" },
    { path: "tests/gpu/test_serve.cpp", label: "16 concurrent against 16 solo, forced preemption" },
  ],
  testing:
    "The 21 scheduler tests run on the CPU against a fake KV cache filled exactly where write_kv would write. They cover admission first come first served up to max_batch, admission stopping at the first request that does not fit, prefill before decode, retiring at max_new_tokens or EOS with the blocks freed, preemption of the most recently admitted request and its recomputation, a sequence that cannot grow preempting itself, preempted sequences returning to the front in admission order, rejected calls changing nothing, and random runs that keep every invariant. Two of them compare whole runs: 16 concurrent requests give the tokens of 16 solo runs, with room and under forced preemption. On the real model the same check runs on the GPU: 16 concurrent greedy requests equal their solo runs, and a 56-block cache that forces preemptions gives every greedy request the output of a roomy one; with sampling at temperature 0.8 the test asks for at least 14 of the 16.",
  excerpts: [
    {
      file: "src/sched/scheduler.cpp",
      from: "    while (!waiting_queue_.empty() &&",
      to: "        tokens_admitted += req_length;",
      caption: "Admission in schedule(): while a slot is free, the front of the queue joins if blocks for its whole length are free; the first request that does not fit stops the loop",
    },
    {
      file: "src/sched/scheduler.cpp",
      from: "        while (!block_manager_.can_append(s)) {",
      to: "            waiting_queue_.push_front(v);",
      caption: "Preemption: while sequence s cannot append, the back of the running list loses its blocks and goes to the front of the waiting queue; s can be its own victim",
    },
    {
      file: "src/kv/block_manager.cpp",
      from: "bool BlockManager::can_append(SeqId seq) const {",
      to: "    return true;",
      caption: "can_append: a sequence needs a free block only when its length is a multiple of the block size",
    },
  ],
  measurements: [
    {
      doc: "results",
      section: load,
      table: 0,
      caption: "the engine with whole-prompt prefill and with token budgets, against vLLM, under the same Poisson arrivals; the preemptions column is 0 in every row",
      command: "scripts/load_sweep.sh, then scripts/load_report.py; scripts/baseline_vllm.py for vLLM",
    },
    {
      doc: "results",
      section: sweep,
      table: 0,
      caption: "max batch 16 to 256, twice as many requests as the max batch submitted at once: the step time, the tokens per second and the most KV blocks ever held",
      command: "scripts/batch_sweep.sh, then scripts/batch_report.py",
    },
    {
      doc: "results",
      section: "throughput-and-latency",
      table: 0,
      caption: "what running requests together buys: output tokens per second against the max batch, 32 requests of 64 prompt and 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --num-requests 32 --prompt-len 64 --output-len 64 --max-batch 16 (and 1, 2, 4, 8; scripts/reproduce.sh)",
    },
  ],
  timeline: [
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
      date: "Nov 3",
      prs: [131],
      branch: "bench/qps-sweep",
      version: "measured",
      author: "claude",
      title: "Requests on their own clock",
      text: `scripts/load_sweep.sh sends 120 requests of mixed lengths as a Poisson process at ${lowRate} to ${highRate} per second, and load_report.py counts goodput: the requests per second that get their first token within ${slo.ttft_ms} ms and their later tokens within ${slo.tpot_ms} ms each. Offline runs submit everything at once and never show a queue.`,
    },
    {
      date: "Nov 3",
      prs: [132],
      branch: "bench/vllm-baseline",
      version: "measured",
      author: "claude",
      title: "vLLM as the online baseline",
      text: `HuggingFace and llama.cpp run static batches, so vLLM is the comparison for arrivals: the same token ids at the same times, the same BF16 weights and the same limit of ${maxBatch} sequences. Its times are taken in an HTTP client; the engine's are taken inside the process.`,
    },
    {
      date: "Nov 5",
      prs: [153],
      branch: "bench/chunked-prefill",
      version: "measured",
      author: "claude",
      title: "Both systems on the same day",
      text: `The sweep that this chapter reads, measured ${run.date} at commit ${run.commit}: the engine's best goodput is ${best.engine.value.toFixed(2)} requests per second with whole-prompt steps and ${best.budget.value.toFixed(2)} with a token budget, vLLM's is ${best.vllm.value.toFixed(2)}, and the engine preempted ${loadPreemptions} times over all its runs.`,
    },
    {
      date: "Nov 9",
      prs: [187],
      branch: "runtime/graph-buckets-wiring",
      version: "measured",
      author: "claude",
      title: "Past 16 rows the step, not the pool, gives out",
      text: `scripts/batch_sweep.sh raises max_batch to ${sweepLast.batch}. The step grows from ${sweepRows[0].tpot} to ${sweepLast.tpot} ms, and ${sweepLast.batch} short requests peak at ${fmt(sweepLast.peak)} of the ${fmt(poolBlocks)} blocks.`,
    },
  ],
};
