export type Config = { numBlocks: number; blockSize: number; maxBatch: number; maxPrefillTokens: number; maxContext: number };

export type Request = { id: number; prompt: number; maxNew: number; output: number; preemptions: number; running: boolean; finished: boolean };

export type Step = { prefill: boolean; seqs: number[] };

export type PrefillChunk = { id: number; first: number; count: number };

export type ChunkedStep = { decode: number[]; prefill: PrefillChunk[]; sampled: number[] };

export const length = (r: Request) => r.prompt + r.output;

export const numTokens = (s: ChunkedStep) => s.decode.length + s.prefill.reduce((n, c) => n + c.count, 0);

export const isEmpty = (s: ChunkedStep) => s.decode.length === 0 && s.prefill.length === 0;

class BlockManager {
  free: number;
  readonly numBlocks: number;
  readonly blockSize: number;
  readonly lengths = new Map<number, number>();
  readonly tables = new Map<number, number>();

  constructor(numBlocks: number, blockSize: number) {
    this.numBlocks = numBlocks;
    this.blockSize = blockSize;
    this.free = numBlocks - 1;
  }

  private blocksFor(tokens: number) {
    return Math.ceil(tokens / this.blockSize);
  }

  canAllocate(tokens: number) {
    return this.free >= this.blocksFor(tokens);
  }

  allocate(id: number, tokens: number) {
    this.free -= this.blocksFor(tokens);
    this.lengths.set(id, tokens);
    this.tables.set(id, this.blocksFor(tokens));
  }

  canAppend(id: number) {
    return this.lengths.get(id)! % this.blockSize !== 0 || this.free >= 1;
  }

  appendSlot(id: number) {
    const len = this.lengths.get(id)!;
    if (len % this.blockSize === 0) {
      this.free--;
      this.tables.set(id, this.tables.get(id)! + 1);
    }
    this.lengths.set(id, len + 1);
  }

  release(id: number) {
    this.free += this.tables.get(id)!;
    this.lengths.delete(id);
    this.tables.delete(id);
  }

  held() {
    return this.numBlocks - 1 - this.free;
  }
}

export class Scheduler {
  readonly config: Config;
  readonly blocks: BlockManager;
  readonly requests = new Map<number, Request>();
  waiting: number[] = [];
  running: number[] = [];
  readonly progress = new Map<number, number>();
  pending = false;
  private sampled: number[] = [];

  constructor(config: Config) {
    if (config.maxBatch < 1) throw new Error("max_batch must be >= 1");
    if (config.maxContext < 1) throw new Error("max_context must be >= 1");
    if (config.maxPrefillTokens < config.maxContext) throw new Error("max_prefill_tokens must be >= max_context");
    if (config.numBlocks - 1 < Math.ceil(config.maxContext / config.blockSize)) throw new Error("num_blocks - 1 must be >= ceil(max_context / block_size)");
    this.config = config;
    this.blocks = new BlockManager(config.numBlocks, config.blockSize);
  }

  submit(id: number, prompt: number, maxNew: number) {
    if (this.requests.has(id)) throw new Error("request id already held");
    this.requests.set(id, { id, prompt, maxNew, output: 0, preemptions: 0, running: false, finished: false });
    this.waiting.push(id);
  }

  private preempt(victim: number) {
    this.blocks.release(victim);
    this.progress.delete(victim);
    const r = this.requests.get(victim)!;
    r.preemptions++;
    r.running = false;
  }

  private makeRoom(s: number) {
    const preempted: number[] = [];
    while (!this.blocks.canAppend(s)) {
      if (this.running.length === 0) break;
      const victim = this.running.pop()!;
      this.preempt(victim);
      preempted.push(victim);
      if (victim === s) break;
    }
    this.waiting.unshift(...preempted.reverse());
    return !preempted.includes(s);
  }

  schedule(): Step {
    if (this.pending) throw new Error("previous non-empty step has not been updated");
    const admitted: number[] = [];
    let admittedTokens = 0;
    while (this.waiting.length > 0 && this.running.length + admitted.length < this.config.maxBatch) {
      const id = this.waiting[0];
      const len = length(this.requests.get(id)!);
      if (!this.blocks.canAllocate(len)) break;
      if (admittedTokens + len > this.config.maxPrefillTokens) break;
      this.waiting.shift();
      this.blocks.allocate(id, len);
      this.requests.get(id)!.running = true;
      admitted.push(id);
      admittedTokens += len;
    }
    if (admitted.length > 0) {
      this.running.push(...admitted);
      this.pending = true;
      this.sampled = [...admitted];
      return { prefill: true, seqs: admitted };
    }
    if (this.running.length === 0) return { prefill: false, seqs: [] };
    const decode: number[] = [];
    let idx = 0;
    while (idx < this.running.length) {
      const s = this.running[idx];
      if (this.makeRoom(s)) {
        this.blocks.appendSlot(s);
        decode.push(s);
        idx++;
      }
    }
    this.pending = true;
    this.sampled = [...decode];
    return { prefill: false, seqs: decode };
  }

  scheduleChunked(tokenBudget: number): ChunkedStep {
    if (tokenBudget < this.config.maxBatch) throw new Error("token_budget must be >= max_batch");
    if (this.pending) throw new Error("previous step has not been updated");
    const decode: number[] = [];
    const prefill: PrefillChunk[] = [];
    const sampled: number[] = [];
    let remaining = tokenBudget;

    let idx = 0;
    while (idx < this.running.length) {
      const s = this.running[idx];
      if (this.progress.has(s)) {
        idx++;
        continue;
      }
      if (this.makeRoom(s)) {
        this.blocks.appendSlot(s);
        decode.push(s);
        sampled.push(s);
        remaining--;
        idx++;
      }
    }

    for (let i = this.running.length - 1; i >= 0; i--) {
      const s = this.running[i];
      const done = this.progress.get(s);
      if (done !== undefined && remaining > 0) {
        const total = length(this.requests.get(s)!);
        const count = Math.min(remaining, total - done);
        prefill.push({ id: s, first: done, count });
        remaining -= count;
        if (done + count === total) {
          this.progress.delete(s);
          sampled.push(s);
        } else {
          this.progress.set(s, done + count);
        }
        break;
      }
    }

    while (this.waiting.length > 0 && this.running.length < this.config.maxBatch && remaining > 0) {
      const id = this.waiting[0];
      const len = length(this.requests.get(id)!);
      if (!this.blocks.canAllocate(len)) break;
      this.waiting.shift();
      this.blocks.allocate(id, len);
      this.requests.get(id)!.running = true;
      this.running.push(id);
      const count = Math.min(remaining, len);
      prefill.push({ id, first: 0, count });
      remaining -= count;
      if (count === len) {
        sampled.push(id);
      } else {
        this.progress.set(id, count);
        break;
      }
    }

    const step = { decode, prefill, sampled };
    if (isEmpty(step)) return step;
    this.pending = true;
    this.sampled = sampled;
    return step;
  }

  update() {
    if (!this.pending) throw new Error("no pending step");
    for (const id of this.sampled) {
      const r = this.requests.get(id)!;
      r.output++;
      if (r.output >= r.maxNew) {
        this.blocks.release(id);
        this.progress.delete(id);
        r.running = false;
        r.finished = true;
        this.running = this.running.filter((x) => x !== id);
      }
    }
    this.pending = false;
    this.sampled = [];
  }
}

export type Mode = "schedule()" | "schedule_chunked()";

export type PoolSize = "roomy" | "tight";

export type Cell = "decode" | "stall" | "chunk" | "first" | "wait" | "preempt";

export type Slice = { id: number; count: number };

export type StepRecord = { decode: number; slices: Slice[]; tokens: number };

export type Lane = { id: number; label: string; kind: "decoder" | "prompt"; prompt: number; from: number; cells: Cell[]; firstAt: number };

export const poolBlocks: Record<PoolSize, number> = { roomy: 513, tight: 177 };
export const decoders = 4;
export const decoderPrompt = 120;
export const decoderNew = 512;
export const dropNew = 32;
export const maxPrompt = 2048;
export const maxBatch = 8;
export const stepCap = 600;

export class Run {
  readonly mode: Mode;
  readonly sched: Scheduler;
  readonly steps: StepRecord[] = [];
  readonly lanes: Lane[] = [];
  readonly stallRun = new Map<number, number>();
  longestStall = 0;
  preemptions = 0;

  constructor(mode: Mode, pool: PoolSize) {
    this.mode = mode;
    this.sched = new Scheduler({ numBlocks: poolBlocks[pool], blockSize: 16, maxBatch, maxPrefillTokens: maxPrompt, maxContext: maxPrompt });
    for (let i = 0; i < decoders; i++) {
      this.sched.submit(i, decoderPrompt, decoderNew);
      this.lanes.push({ id: i, label: String.fromCharCode(65 + i), kind: "decoder", prompt: decoderPrompt, from: 0, cells: [], firstAt: 0 });
    }
    this.sched.schedule();
    this.sched.update();
  }

  drop(id: number, prompt: number) {
    this.sched.submit(id, prompt, dropNew);
    this.lanes.push({ id, label: `P${id - decoders + 1}`, kind: "prompt", prompt, from: this.steps.length, cells: [], firstAt: -1 });
  }

  private plan(budget: number): { decode: number[]; slices: Slice[]; sampled: number[] } {
    if (this.mode === "schedule_chunked()") {
      const step = this.sched.scheduleChunked(budget);
      return { decode: step.decode, slices: step.prefill.map((c) => ({ id: c.id, count: c.count })), sampled: step.sampled };
    }
    const step = this.sched.schedule();
    if (step.prefill) return { decode: [], slices: step.seqs.map((id) => ({ id, count: length(this.sched.requests.get(id)!) })), sampled: step.seqs };
    return { decode: step.seqs, slices: [], sampled: step.seqs };
  }

  advance(budget: number) {
    const reqs = this.sched.requests;
    const before = new Map([...reqs].map((e) => [e[0], { output: e[1].output, preemptions: e[1].preemptions, finished: e[1].finished }]));
    const { decode, slices, sampled } = this.plan(budget);
    const tokens = decode.length + slices.reduce((n, c) => n + c.count, 0);
    if (tokens > 0) this.sched.update();
    this.steps.push({ decode: decode.length, slices, tokens });
    for (const lane of this.lanes) {
      const was = before.get(lane.id)!;
      if (was.finished) continue;
      const r = reqs.get(lane.id)!;
      const preempted = r.preemptions > was.preemptions;
      this.preemptions += r.preemptions - was.preemptions;
      const got = sampled.includes(lane.id);
      let cell: Cell;
      if (decode.includes(lane.id)) cell = "decode";
      else if (slices.some((c) => c.id === lane.id)) cell = got && was.output === 0 ? "first" : "chunk";
      else if (preempted) cell = "preempt";
      else cell = was.output >= 1 ? "stall" : "wait";
      lane.cells.push(cell);
      if (got && was.output === 0 && lane.firstAt < 0) lane.firstAt = this.steps.length;
      const run = was.output >= 1 && !got ? (this.stallRun.get(lane.id) ?? 0) + 1 : 0;
      this.stallRun.set(lane.id, run);
      this.longestStall = Math.max(this.longestStall, run);
    }
  }

  settled() {
    return this.lanes.every((l) => l.kind === "decoder" || l.firstAt >= 0) || this.steps.length >= stepCap;
  }

  stats() {
    const biggest = this.steps.reduce((m, s) => Math.max(m, s.tokens), 0);
    const total = this.steps.reduce((n, s) => n + s.tokens, 0);
    const prompts = this.lanes.filter((l) => l.kind === "prompt");
    const last = prompts[prompts.length - 1];
    const ttft = last ? { steps: last.firstAt >= 0 ? last.firstAt - last.from : this.steps.length - last.from, done: last.firstAt >= 0, prompt: last.prompt } : null;
    return {
      steps: this.steps.length,
      longestStall: this.longestStall,
      biggest,
      perStep: this.steps.length ? total / this.steps.length : 0,
      ttft,
      preemptions: this.preemptions,
      waiting: this.sched.waiting.length,
      running: this.sched.running.length,
      blocksHeld: this.sched.blocks.held(),
    };
  }
}

export class Playground {
  readonly pool: PoolSize;
  readonly runs: Record<Mode, Run>;
  nextId = decoders;

  constructor(pool: PoolSize) {
    this.pool = pool;
    this.runs = { "schedule()": new Run("schedule()", pool), "schedule_chunked()": new Run("schedule_chunked()", pool) };
  }

  drop(prompt: number) {
    const n = Math.max(1, Math.min(maxPrompt, Math.round(prompt)));
    const id = this.nextId++;
    this.runs["schedule()"].drop(id, n);
    this.runs["schedule_chunked()"].drop(id, n);
  }

  advance(budget: number) {
    this.runs["schedule()"].advance(budget);
    this.runs["schedule_chunked()"].advance(budget);
  }

  settled() {
    return this.runs["schedule()"].settled() && this.runs["schedule_chunked()"].settled();
  }

  steps() {
    return this.runs["schedule()"].steps.length;
  }
}

export function predict(pool: PoolSize, prompt: number, budget: number) {
  const g = new Playground(pool);
  g.drop(prompt);
  while (!g.settled()) g.advance(budget);
  return { legacy: g.runs["schedule()"].stats(), chunked: g.runs["schedule_chunked()"].stats() };
}
