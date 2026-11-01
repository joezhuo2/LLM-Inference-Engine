import type { TraceStep } from "../trace";

export const FREE = -1;
export const UNKNOWN = -2;
export const RESERVED = -3;

export type Seq = {
  id: number;
  slot: number;
  prompt: number | null;
  length: number;
  exact: boolean;
  blocks: number[];
  since: number[];
  outputs: string[];
  text: string;
  admitted: number;
  finished: number;
  preemptions: number;
  running: boolean;
};

export type ModelOptions = { totalBlocks: number; blockSize: number; slots: number; prompts?: Map<number, number>; keepSteps?: number };

export class DashModel {
  readonly totalBlocks: number;
  readonly blockSize: number;
  readonly slots: number;
  readonly owner: Int32Array;
  readonly seqs = new Map<number, Seq>();
  readonly steps: TraceStep[] = [];
  readonly stepSlots: number[][] = [];
  readonly gaps = new Map<number, number>();
  firstIndex = 0;
  lastIndex = -1;
  missed = 0;
  used = 0;
  unknown = 0;
  highWater = 0;
  private readonly prompts: Map<number, number> | undefined;
  private readonly keepSteps: number;
  private readonly slotOwner: (number | null)[];
  private lowestFree = 1;
  private grown: Seq[] = [];
  private current = -1;
  private synced = false;

  constructor({ totalBlocks, blockSize, slots, prompts, keepSteps = Infinity }: ModelOptions) {
    this.totalBlocks = totalBlocks;
    this.blockSize = blockSize;
    this.slots = slots;
    this.prompts = prompts;
    this.keepSteps = keepSteps;
    this.owner = new Int32Array(totalBlocks).fill(FREE);
    this.owner[0] = RESERVED;
    this.slotOwner = Array.from({ length: slots }, () => null);
  }

  get last(): TraceStep | undefined {
    return this.steps[this.steps.length - 1];
  }

  private position(index: number): number {
    let lo = 0;
    let hi = this.steps.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.steps[mid].index < index) lo = mid + 1;
      else hi = mid;
    }
    return this.steps[lo]?.index === index ? lo : -1;
  }

  step(index: number): TraceStep | undefined {
    const at = this.position(index);
    return at < 0 ? undefined : this.steps[at];
  }

  slotsOf(index: number): number[] | undefined {
    const at = this.position(index);
    return at < 0 ? undefined : this.stepSlots[at];
  }

  free(): number {
    return this.totalBlocks - 1 - this.used;
  }

  apply(step: TraceStep) {
    if (this.lastIndex >= 0 && step.index !== this.lastIndex + 1) this.markGap(step.index);
    if (this.lastIndex < 0) {
      this.firstIndex = step.index;
      this.synced = step.index === 0;
    }
    this.current = step.index;
    const fresh = step.kind === "prefill" ? this.prefill(step) : this.decode(step);
    step.seqs.forEach((id, i) => this.addToken(this.seqs.get(id)!, step.tokens[i]?.text ?? ""));
    this.record(step);
    for (const id of step.finished) this.finish(id, step.index);
    this.reconcile(step, fresh);
    this.synced = true;
    this.lastIndex = step.index;
  }

  private markGap(index: number) {
    const missing = index - this.lastIndex - 1;
    if (missing > 0) {
      this.missed += missing;
      this.gaps.set(index, missing);
    }
    this.synced = false;
  }

  private record(step: TraceStep) {
    this.steps.push(step);
    this.stepSlots.push(step.seqs.map((id) => this.seqs.get(id)!.slot));
    if (this.steps.length > this.keepSteps) {
      this.steps.shift();
      this.stepSlots.shift();
      this.gaps.delete(this.firstIndex);
      this.firstIndex = this.steps[0].index;
    }
  }

  private seqFor(id: number): Seq {
    let seq = this.seqs.get(id);
    if (!seq) {
      const prompt = this.prompts?.get(id) ?? null;
      seq = { id, slot: -1, prompt, length: prompt ?? 0, exact: prompt !== null, blocks: [], since: [], outputs: [], text: "", admitted: -1, finished: -1, preemptions: 0, running: false };
      this.seqs.set(id, seq);
    }
    return seq;
  }

  private start(seq: Seq, index: number) {
    if (seq.admitted < 0) seq.admitted = index;
    seq.running = true;
    if (seq.slot < 0) {
      const slot = this.slotOwner.indexOf(null);
      if (slot >= 0) this.slotOwner[slot] = seq.id;
      seq.slot = slot;
    }
  }

  private stop(seq: Seq) {
    this.release(seq);
    seq.running = false;
    if (seq.slot >= 0) this.slotOwner[seq.slot] = null;
    seq.slot = -1;
  }

  private prefill(step: TraceStep): Seq[] {
    const fresh: Seq[] = [];
    for (const id of step.seqs) {
      const seq = this.seqFor(id);
      if (seq.running) {
        this.stop(seq);
        seq.preemptions++;
      }
      this.start(seq, step.index);
      if (seq.exact) {
        seq.length = (seq.prompt ?? 0) + seq.outputs.length;
        for (let b = 0; b < Math.ceil(seq.length / this.blockSize); b++) this.take(seq);
      } else {
        fresh.push(seq);
      }
    }
    return fresh;
  }

  private decode(step: TraceStep): Seq[] {
    const inBatch = new Set(step.seqs);
    for (const seq of this.seqs.values()) {
      if (seq.running && !inBatch.has(seq.id)) {
        this.stop(seq);
        seq.preemptions++;
      }
    }
    const fresh: Seq[] = [];
    this.grown = [];
    for (const id of step.seqs) {
      const seq = this.seqFor(id);
      if (!seq.running) this.start(seq, step.index);
      if (seq.exact && seq.blocks.length > 0) {
        if (seq.length % this.blockSize === 0) {
          this.take(seq);
          if (seq.prompt === null) this.grown.push(seq);
        }
        seq.length++;
      } else if (seq.blocks.length > 0) {
        seq.length++;
        fresh.push(seq);
      }
    }
    return fresh;
  }

  private addToken(seq: Seq, text: string) {
    seq.outputs.push(text);
    seq.text += text;
  }

  private finish(id: number, index: number) {
    const seq = this.seqs.get(id);
    if (!seq) return;
    this.stop(seq);
    seq.finished = index;
  }

  private reconcile(step: TraceStep, fresh: Seq[]) {
    let diff = this.free() - step.free_blocks;
    if (diff > 0 && this.synced) diff = step.kind === "prefill" ? this.splitPrefill(fresh, diff) : this.growDecode(fresh, diff);
    for (const seq of fresh) {
      const capacity = seq.blocks.length * this.blockSize;
      if (seq.length > capacity) seq.length = capacity;
    }
    while (diff > 0) {
      this.takeUnknown();
      diff--;
    }
    if (diff < 0) diff += this.freeUnknown(-diff);
    if (diff < 0) this.trim(-diff);
  }

  private trim(count: number) {
    const slack = (s: Seq) => s.length - s.blocks.length * this.blockSize;
    const others = this.liveSeqs()
      .filter((s) => s.prompt === null && !this.grown.includes(s) && s.blocks.length > 0)
      .sort((a, b) => slack(a) - slack(b));
    for (const seq of [...this.grown, ...others]) {
      if (count === 0) return;
      this.freeBlock(seq.blocks.pop()!);
      seq.since.pop();
      seq.exact = false;
      seq.length = Math.min(seq.length, seq.blocks.length * this.blockSize);
      count--;
    }
  }

  private splitPrefill(fresh: Seq[], diff: number): number {
    const live = fresh.filter((s) => s.running);
    if (live.length === 0) return diff;
    const share = Math.floor(diff / live.length);
    live.forEach((seq, i) => {
      const n = share + (i < diff % live.length ? 1 : 0);
      for (let b = 0; b < n; b++) this.take(seq);
      seq.length = Math.max(0, n * this.blockSize - this.blockSize / 2);
    });
    return 0;
  }

  private growDecode(fresh: Seq[], diff: number): number {
    const slack = (s: Seq) => s.length - s.blocks.length * this.blockSize;
    const ranked = fresh.filter((s) => s.running && (this.unknown === 0 || slack(s) > -this.blockSize / 4)).sort((a, b) => slack(b) - slack(a));
    for (const seq of ranked.slice(0, diff)) {
      seq.length = seq.blocks.length * this.blockSize + 1;
      this.take(seq);
      seq.exact = true;
      diff--;
    }
    return diff;
  }

  private takeBlock(owner: number): number {
    let id = this.lowestFree;
    while (id < this.totalBlocks && this.owner[id] !== FREE) id++;
    if (id >= this.totalBlocks) return -1;
    this.owner[id] = owner;
    this.lowestFree = id + 1;
    this.used++;
    this.highWater = Math.max(this.highWater, id);
    return id;
  }

  private take(seq: Seq) {
    const id = this.takeBlock(seq.id);
    if (id < 0) return;
    seq.blocks.push(id);
    seq.since.push(this.current);
  }

  private takeUnknown() {
    if (this.takeBlock(UNKNOWN) >= 0) this.unknown++;
  }

  private freeBlock(id: number) {
    this.owner[id] = FREE;
    this.used--;
    if (id < this.lowestFree) this.lowestFree = id;
  }

  private release(seq: Seq) {
    for (const id of seq.blocks) this.freeBlock(id);
    seq.blocks = [];
    seq.since = [];
  }

  private freeUnknown(count: number): number {
    let freed = 0;
    for (let id = this.totalBlocks - 1; id > 0 && freed < count && this.unknown > 0; id--) {
      if (this.owner[id] !== UNKNOWN) continue;
      this.freeBlock(id);
      this.unknown--;
      freed++;
    }
    return freed;
  }

  filled(seq: Seq, logical: number): number {
    return Math.max(0, Math.min(this.blockSize, seq.length - logical * this.blockSize));
  }

  occupant(slot: number): number | null {
    return this.slotOwner[slot] ?? null;
  }

  liveSeqs(): Seq[] {
    return [...this.seqs.values()].filter((s) => s.running);
  }
}
