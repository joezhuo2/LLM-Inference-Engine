import { type KernelStory, measured } from "./types";

const prefill = "prefill-against-prompt-length";
const load = "under-load";

export const prefillMs = (system: string, prompt: string) => measured(prefill, { system }, prompt, 0, "results");
export const goodput = (system: string, qps: string) => measured(load, { system, QPS: qps }, "goodput req/s", 0, "results");

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
  title: "chunked prefill",
  short: "A token budget for every step: each decoding sequence gets its row first and the rest goes to the one prompt in prefill, so a long prompt spreads over several steps instead of stopping the others.",
  formula: "chunk = min(budget − decode rows, prompt tokens left)",
  headline: {
    value: `${(prefillMs("engine", "2,048") / 1000).toFixed(1)} s`,
    label: "to prefill one 2,048-token prompt alone, while every running sequence waits",
  },
  sandbox: {
    title: "play with the budget",
    intro: "A TypeScript port of Scheduler::schedule() and Scheduler::schedule_chunked() in src/sched/scheduler.cpp runs the same arrivals under both rules: four sequences already decoding, a pool of 16-token KV blocks, first come first served, decode rows before prefill, preemption from the back of the running list. Drop prompts, pick the budget, and watch which decoders lose their row. The model counts steps and tokens, it does not time anything. Choose the tight pool and a small budget to see the prompt in prefill preempted and started again.",
  },
  sources: [
    { path: "src/sched/scheduler.cpp", label: "Scheduler::schedule_chunked" },
    { path: "include/engine/sched/scheduler.h", label: "PrefillChunk, ChunkedStep" },
  ],
  tests: [{ path: "tests/unit/test_scheduler_chunked.cpp", label: "25 scheduler tests" }],
  testing:
    "25 unit tests, on the CPU: the budget limits (a budget below the batch size is refused, a step never exceeds it, a step with unfinished prefill uses all of it), decode rows first, a long prompt spread over steps with chunks that start at 0 and stay contiguous, the sampled list, first_token stamped when the token arrives, preemption of a partly prefilled sequence and the replay of a preempted decoder, cancel, and random runs with preemption whose answers must equal each request alone for any budget. Nothing calls it yet: the runner wiring and the measurement of its effect come in the next branch.",
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
      caption: "the engine against vLLM under Poisson arrivals; the cause of the gap is not measured on its own",
      command: "scripts/load_sweep.sh",
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
      text: `Goodput of ${goodput("vLLM", "6").toFixed(1)} requests per second against the engine's ${goodput("engine + graphs", "4").toFixed(1)}. vLLM mixes prefill chunks into decode steps; whether stalls explain the gap has not been measured on their own.`,
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
      text: "Joe implemented the budgeted step. A fix drafted with Claude made a preempted decoder replay its prompt and output, and the tests moved into unit_tests. schedule() is unchanged and nothing calls the new method yet.",
    },
  ],
};
