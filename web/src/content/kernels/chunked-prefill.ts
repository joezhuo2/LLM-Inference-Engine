import { type KernelStory, measured } from "./types";

const prefill = "prefill-against-prompt-length";
const load = "under-load";
const stall = "a-long-prompt-among-running-requests";

export const chunkBudget = "256";
export const chunked = `engine + graphs, budget ${chunkBudget}`;

export const prefillMs = (system: string, prompt: string) => measured(prefill, { system }, prompt, 0, "results");
export const goodput = (system: string, qps: string) => measured(load, { system, QPS: qps }, "goodput req/s", 0, "results");
export const ttftTail = (system: string, qps: string) => measured(load, { system, QPS: qps }, "TTFT p99 ms", 0, "results");
export const largestGap = (system: string, qps: string) => measured(load, { system, QPS: qps }, "largest gap ms", 1, "results");
export const medianGap = (system: string, qps: string) => measured(load, { system, QPS: qps }, "gap p50 ms", 1, "results");
export const stallMs = (budget: string, column: "TTFT of the long prompt ms" | "largest gap ms") => measured(stall, { "token budget": budget }, column, 0, "results");
export const stallPrompt = measured(stall, { "token budget": "none" }, "long prompt tokens", 0, "results");
export const stallDecoding = measured(stall, { "token budget": "none" }, "decoding when it arrives", 0, "results");

export function peakGoodput(system: string): { qps: string; value: number } {
  return ["1", "2", "3", "4", "5", "6", "8"].map((qps) => ({ qps, value: goodput(system, qps) })).reduce((a, b) => (b.value > a.value ? b : a));
}

export const fmt = (v: number) => v.toLocaleString("en-US");

export type ChunkStep = { decode: number; chunk: number };

export function chunkedSteps(budget: number, decoders: number, prompt: number): ChunkStep[] {
  const steps: ChunkStep[] = [];
  for (let done = 0; done < prompt; ) {
    const chunk = Math.min(budget - decoders, prompt - done);
    steps.push({ decode: decoders, chunk });
    done += chunk;
  }
  return steps;
}

export const chunkedPrefill: KernelStory = {
  slug: "chunked-prefill",
  index: 11,
  book: true,
  about: true,
  title: "chunked prefill",
  short: "A token budget for every step: each decoding sequence gets its row first and the rest goes to the one prompt in prefill, so a long prompt spreads over several steps instead of stopping the others.",
  formula: "chunk = min(budget − decode rows, prompt tokens left)",
  headline: {
    value: `${fmt(stallMs("none", "largest gap ms"))} → ${fmt(stallMs(chunkBudget, "largest gap ms"))} ms`,
    label: `the longest wait between two tokens of a running request when a ${fmt(stallPrompt)}-token prompt arrives, without and with a budget of ${chunkBudget}`,
  },
  sandbox: {
    title: "play with the budget",
    intro: "A TypeScript port of Scheduler::schedule() and Scheduler::schedule_chunked() in src/sched/scheduler.cpp runs the same arrivals under both rules: four sequences already decoding, a pool of 16-token KV blocks, first come first served, decode rows before prefill, preemption from the back of the running list. Drop prompts, pick the budget, and watch which decoders lose their row. The model counts steps and tokens, it does not time anything. Choose the tight pool and a small budget to see the prompt in prefill preempted and started again.",
  },
  sources: [
    { path: "src/sched/scheduler.cpp", label: "Scheduler::schedule_chunked" },
    { path: "include/engine/sched/scheduler.h", label: "PrefillChunk, ChunkedStep" },
    { path: "src/sched/paged_batch.cpp", label: "build_chunked_batch" },
    { path: "src/runtime/serve.cpp", label: "serve_step with a token budget" },
  ],
  tests: [
    { path: "tests/unit/test_scheduler_chunked.cpp", label: "25 scheduler tests" },
    { path: "tests/gpu/test_serve.cpp", label: "outputs under a budget" },
  ],
  testing:
    "25 unit tests, on the CPU: the budget limits (a budget below the batch size is refused, a step never exceeds it, a step with unfinished prefill uses all of it), decode rows first, a long prompt spread over steps with chunks that start at 0 and stay contiguous, the sampled list, first_token stamped when the token arrives, preemption of a partly prefilled sequence and the replay of a preempted decoder, cancel, and random runs with preemption whose answers must equal each request alone for any budget. On the GPU, serve_step runs it under --token-budget: greedy outputs of the 16 reference prompts at budgets from 16 to 2,048 equal whole-prompt prefill or first differ at a tie within 2 BF16 ulps of the best logit, with and without forced preemption, and chunked steps match the contiguous runner row by row.",
  excerpts: [
    {
      file: "src/sched/scheduler.cpp",
      from: "int total = static_cast<int>(requests_.at(s).length());",
      to: "break;",
      caption: "What is left of the budget goes to the sequence in prefill: a chunk from where it stopped, and only the chunk that reaches the end adds the sequence to the sampled list",
    },
    {
      file: "src/sched/scheduler.cpp",
      from: "if (prefill_progress_.count(s)) {",
      to: "preempted_this_step.push_back(victim);",
      caption: "Decode rows come first and skip the sequence in prefill; when a decoder cannot append, the victim is the back of the running list, which is the partly prefilled one",
    },
  ],
  measurements: [
    {
      doc: "results",
      section: prefill,
      table: 0,
      caption: "TTFT in ms against prompt tokens, one request per run: the time every running sequence waits under schedule()",
      command: "scripts/prefill_ttft.py --engine build/release/bin/engine --model models/tinyllama",
    },
    {
      doc: "results",
      section: load,
      table: 0,
      caption: "the engine with whole-prompt prefill and with token budgets of 256, 512 and 2,048, against vLLM, under the same Poisson arrivals",
      command: "TOKEN_BUDGET=256 scripts/load_sweep.sh, then scripts/load_report.py",
    },
    {
      doc: "results",
      section: load,
      table: 1,
      caption: "gaps between two tokens of one request, from the step traces of the same runs: the stall a long prefill step causes",
      command: "TOKEN_BUDGET=256 TRACES=<dir> scripts/load_sweep.sh, then scripts/load_report.py --traces <dir>",
    },
    {
      doc: "results",
      section: stall,
      table: 0,
      caption: "one long prompt arriving among short requests that are decoding, at each budget",
      command: "scripts/stall_probe.py --budgets 0 256 512 2048",
    },
  ],
  timeline: [
    {
      date: "Nov 3",
      prs: [130],
      branch: "bench/prefill-ttft",
      version: "measured",
      author: "claude",
      title: "A long prompt is slow, and so is everyone waiting",
      text: `One request per run: TTFT grows from ${prefillMs("engine", "128")} ms at 128 tokens to ${prefillMs("engine", "2,048")} ms at 2,048. Under schedule() the prompt is one step, and no running sequence decodes until it ends.`,
    },
    {
      date: "Nov 3",
      prs: [132],
      branch: "bench/vllm-baseline",
      version: "measured",
      author: "claude",
      title: "vLLM on the same arrivals",
      text: "vLLM served the same token ids at the same arrival times and kept its tail TTFT far lower. It mixes prefill chunks into decode steps with a budget of 2,048 tokens; the engine's steps were all prefill or all decode, so prefill stalls were the suspect.",
    },
    {
      date: "Nov 4",
      prs: [133],
      branch: "sched/chunked-prefill-stub",
      version: "spec",
      author: "claude",
      title: "The spec and 25 tests",
      text: "schedule_chunked written as a TODO(JOE) spec with PrefillChunk and ChunkedStep, and 25 tests that a throwaway implementation passes and 14 deliberate mutants of it fail.",
    },
    {
      date: "Nov 4",
      prs: [134],
      branch: "sched/chunked-prefill-impl",
      version: "scheduler",
      author: "both",
      title: "Decode rows first, then the prompt",
      text: "Joe implemented the budgeted step. A fix drafted with Claude made a preempted decoder replay its prompt and output, and the tests moved into unit_tests. schedule() is unchanged.",
    },
    {
      date: "Nov 4",
      prs: [149],
      branch: "sched/chunked-prefill-wiring",
      version: "wired",
      author: "claude",
      title: "The runner takes the budget",
      text: "build_chunked_batch puts decode rows first and chunk rows after, with logit rows only for sampled sequences. A step of unfinished chunks writes KV and samples nothing, pure decode steps keep their CUDA Graph, and engine bench and engine serve take --token-budget.",
    },
    {
      date: "Nov 5",
      prs: [],
      branch: "bench/chunked-prefill",
      version: "measured",
      author: "claude",
      title: "What the budget buys",
      text: `At 4 requests per second a budget of ${chunkBudget} cuts the longest wait between two tokens from ${largestGap("engine + graphs", "4")} to ${largestGap(chunked, "4")} ms and TTFT p99 from ${ttftTail("engine + graphs", "4")} to ${ttftTail(chunked, "4")} ms. Peak goodput moves from ${peakGoodput("engine + graphs").value.toFixed(2)} to ${peakGoodput(chunked).value.toFixed(2)} requests per second; vLLM reaches ${peakGoodput("vLLM").value.toFixed(2)}, so stalls were only part of the gap.`,
    },
  ],
};
