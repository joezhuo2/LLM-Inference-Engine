export type Mode = "paged" | "contiguous";

export type Shape = { prompt: number; output: number; reserve: number };

export type Seq = { id: number; kind: number; generated: number; slots: number; table: number[]; preemptions: number; admittedAt: number; finishedAt: number };

export type Cell = { seq: number; kind: number; logical: number; filled: number; reserved: boolean };

export class Pool {
  readonly free: boolean[];
  readonly seqs = new Map<number, Seq>();
  waiting: number[] = [];
  running: number[] = [];
  preemptions = 0;
  steps = 0;

  readonly mode: Mode;
  readonly numBlocks: number;
  readonly blockSize: number;
  readonly shapes: Shape[];

  constructor(mode: Mode, numBlocks: number, blockSize: number, shapes: Shape[]) {
    this.mode = mode;
    this.numBlocks = numBlocks;
    this.blockSize = blockSize;
    this.shapes = shapes;
    this.free = Array.from({ length: numBlocks }, (_, i) => i > 0);
  }

  add(id: number, kind: number) {
    this.seqs.set(id, { id, kind, generated: 0, slots: 0, table: [], preemptions: 0, admittedAt: -1, finishedAt: -1 });
    this.waiting.push(id);
  }

  length(s: Seq) {
    return this.shapes[s.kind].prompt + s.generated;
  }

  blocksFor(tokens: number) {
    return Math.ceil(tokens / this.blockSize);
  }

  numFree() {
    return this.free.reduce((n, f) => n + (f ? 1 : 0), 0);
  }

  longestRun() {
    let best = 0;
    let run = 0;
    for (const f of this.free) {
      run = f ? run + 1 : 0;
      best = Math.max(best, run);
    }
    return best;
  }

  need(s: Seq) {
    return this.mode === "paged" ? this.blocksFor(this.length(s)) : this.blocksFor(this.shapes[s.kind].reserve);
  }

  canAllocate(s: Seq) {
    return this.mode === "paged" ? this.numFree() >= this.need(s) : this.longestRun() >= this.need(s);
  }

  private take() {
    const id = this.free.indexOf(true, 1);
    this.free[id] = false;
    return id;
  }

  private allocate(s: Seq) {
    if (this.mode === "paged") {
      s.table = Array.from({ length: this.need(s) }, () => this.take());
    } else {
      const need = this.need(s);
      let run = 0;
      let start = 0;
      for (let i = 1; i < this.numBlocks; i++) {
        run = this.free[i] ? run + 1 : 0;
        if (run === need) {
          start = i - need + 1;
          break;
        }
      }
      s.table = Array.from({ length: need }, (_, j) => start + j);
      for (const b of s.table) this.free[b] = false;
    }
    s.slots = this.length(s);
  }

  private release(s: Seq) {
    for (const b of s.table) this.free[b] = true;
    s.table = [];
    s.slots = 0;
  }

  private canAppend(s: Seq) {
    return s.slots < s.table.length * this.blockSize || this.numFree() > 0;
  }

  private finishStep(ids: number[]) {
    for (const id of ids) {
      const s = this.seqs.get(id)!;
      s.generated++;
      if (s.generated >= this.shapes[s.kind].output) {
        this.release(s);
        s.finishedAt = this.steps;
        this.running = this.running.filter((r) => r !== id);
      }
    }
  }

  step(): "prefill" | "decode" | "idle" {
    this.steps++;
    const admitted: number[] = [];
    while (this.waiting.length) {
      const s = this.seqs.get(this.waiting[0])!;
      if (!this.canAllocate(s)) break;
      this.waiting.shift();
      this.allocate(s);
      if (s.admittedAt < 0) s.admittedAt = this.steps;
      admitted.push(s.id);
    }
    if (admitted.length) {
      this.running.push(...admitted);
      this.finishStep(admitted);
      return "prefill";
    }
    if (!this.running.length) return "idle";
    const decoded: number[] = [];
    let i = 0;
    while (i < this.running.length) {
      const s = this.seqs.get(this.running[i])!;
      let preempted = false;
      const victims: number[] = [];
      while (!this.canAppend(s)) {
        const victim = this.running.pop()!;
        const v = this.seqs.get(victim)!;
        this.release(v);
        v.preemptions++;
        this.preemptions++;
        victims.push(victim);
        if (victim === s.id) {
          preempted = true;
          break;
        }
      }
      this.waiting.unshift(...victims.reverse());
      if (preempted) continue;
      if (s.slots === s.table.length * this.blockSize) s.table.push(this.take());
      s.slots++;
      decoded.push(s.id);
      i++;
    }
    this.finishStep(decoded);
    return "decode";
  }

  cells(): (Cell | null)[] {
    const out: (Cell | null)[] = Array.from({ length: this.numBlocks }, () => null);
    for (const id of this.running) {
      const s = this.seqs.get(id)!;
      s.table.forEach((b, j) => {
        const filled = Math.max(0, Math.min(this.blockSize, s.slots - j * this.blockSize));
        out[b] = { seq: id, kind: s.kind, logical: j, filled, reserved: true };
      });
    }
    return out;
  }

  totals() {
    let written = 0;
    let held = 0;
    for (const id of this.running) {
      const s = this.seqs.get(id)!;
      written += s.slots;
      held += s.table.length;
    }
    return { running: this.running.length, waiting: this.waiting.length, held, written, empty: held * this.blockSize - written, preemptions: this.preemptions };
  }

  busy() {
    return this.running.length > 0 || this.waiting.length > 0;
  }
}
