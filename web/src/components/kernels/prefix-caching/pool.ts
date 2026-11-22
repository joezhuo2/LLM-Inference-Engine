type Meta = { ref: number; published: boolean; key: string };

export class MiniPool {
  readonly blocks: Meta[];
  readonly lru: number[] = [];
  readonly tables = new Map<number, number[]>();
  readonly evicted: number[] = [];
  private readonly map = new Map<string, number>();
  private readonly blank: boolean[];
  readonly block: number;

  constructor(size: number, block: number) {
    this.block = block;
    this.blocks = Array.from({ length: size }, () => ({ ref: 0, published: false, key: "" }));
    this.blank = Array.from({ length: size }, (_, i) => i > 0);
  }

  freeCount() {
    return this.blank.filter(Boolean).length + this.lru.length;
  }

  matchBlocks(tokens: number[]): number[] {
    const out: number[] = [];
    for (let b = 0; b < Math.floor((tokens.length - 1) / this.block); b++) {
      const id = this.map.get(tokens.slice(0, (b + 1) * this.block).join(","));
      if (id === undefined) break;
      out.push(id);
    }
    return out;
  }

  canAllocateTokens(tokens: number[]) {
    const matched = this.matchBlocks(tokens);
    const cached = matched.filter((id) => this.blocks[id].ref === 0).length;
    return this.freeCount() - cached >= Math.ceil(tokens.length / this.block) - matched.length;
  }

  private allocateBlock() {
    let id = this.blank.indexOf(true);
    if (id < 0) {
      id = this.lru.shift() ?? -1;
      if (id < 0) throw new Error("MiniPool: out of blocks despite the check");
      this.map.delete(this.blocks[id].key);
      this.blocks[id].published = false;
      this.blocks[id].key = "";
      this.evicted.push(id);
    }
    this.blank[id] = false;
    this.blocks[id].ref = 1;
    return id;
  }

  allocateTokens(seq: number, tokens: number[]) {
    if (!this.canAllocateTokens(tokens)) throw new Error("MiniPool: no room");
    const matched = this.matchBlocks(tokens);
    for (const id of matched) {
      if (this.blocks[id].ref === 0) this.lru.splice(this.lru.indexOf(id), 1);
      this.blocks[id].ref++;
    }
    const table = [...matched];
    while (table.length < Math.ceil(tokens.length / this.block)) table.push(this.allocateBlock());
    this.tables.set(seq, table);
    return matched.length * this.block;
  }

  publish(seq: number, tokens: number[]) {
    const table = this.tables.get(seq) ?? [];
    for (let b = 0; b < Math.floor(tokens.length / this.block); b++) {
      const meta = this.blocks[table[b]];
      const key = tokens.slice(0, (b + 1) * this.block).join(",");
      if (meta.published || this.map.has(key)) continue;
      this.map.set(key, table[b]);
      meta.published = true;
      meta.key = key;
    }
  }

  free(seq: number) {
    for (const id of [...(this.tables.get(seq) ?? [])].reverse()) {
      const meta = this.blocks[id];
      if (--meta.ref > 0) continue;
      if (meta.published) this.lru.push(id);
      else this.blank[id] = true;
    }
    this.tables.delete(seq);
  }
}

export type PoolKnobs = { shared: number; requests: number; pool: number };
export type PoolSlot = { state: "blank" | "held" | "cached" | "evicted"; count: number };
export type PoolRow = { label: string; blocks: number[]; shared: number; reused: number; prefilled: number; state: "running" | "finished" | "waiting"; need: number };
export type PoolPhase = { slots: PoolSlot[]; rows: PoolRow[]; reused: number; prefilled: number; evicted: number[]; held: number; cached: number; blank: number; late: number };

export const poolBlock = 16;
export const poolOwn = 8;

export function poolRun(k: PoolKnobs): PoolPhase[] {
  const p = new MiniPool(k.pool + 1, poolBlock);
  const start = Array.from({ length: k.shared * poolBlock }, (_, i) => 1000 + i);
  const rows: PoolRow[] = [];
  let reused = 0;
  let prefilled = 0;
  let late = 0;
  const phases: PoolPhase[] = [];
  const prompts = new Map<number, number[]>();
  const admit = (seq: number, label: string, tokens: number[]) => {
    prompts.set(seq, tokens);
    const need = Math.ceil(tokens.length / poolBlock) - p.matchBlocks(tokens).length;
    if (!p.canAllocateTokens(tokens)) {
      rows[seq] = { label, blocks: [], shared: 0, reused: 0, prefilled: 0, state: "waiting", need };
      return false;
    }
    const cached = p.allocateTokens(seq, tokens);
    p.publish(seq, tokens);
    reused += cached;
    prefilled += tokens.length - cached;
    rows[seq] = { label, blocks: [...(p.tables.get(seq) ?? [])], shared: cached / poolBlock, reused: cached, prefilled: tokens.length - cached, state: "running", need };
    return true;
  };
  const finish = (seq: number) => {
    if (!p.tables.has(seq)) return;
    p.free(seq);
    rows[seq].state = "finished";
  };
  const snap = (since: number) => {
    const evicted = p.evicted.slice(since);
    const slots = p.blocks.slice(1).map((m, i): PoolSlot => ({ state: m.ref > 0 ? (evicted.includes(i + 1) ? "evicted" : "held") : m.published ? "cached" : "blank", count: m.ref }));
    phases.push({ slots, rows: rows.map((r) => ({ ...r })), reused, prefilled, evicted, held: slots.filter((s) => s.count > 0).length, cached: p.lru.length, blank: p.freeCount() - p.lru.length, late });
  };
  for (let r = 0; r < k.requests; r++) admit(r, `request ${r + 1}`, [...start, ...Array.from({ length: poolOwn }, (_, i) => 100 * (r + 1) + i)]);
  snap(0);
  const running = rows.flatMap((r, seq) => (r.state === "running" ? [seq] : []));
  const queue = rows.flatMap((r, seq) => (r.state === "waiting" ? [seq] : []));
  while (running.length) {
    finish(running.shift()!);
    while (queue.length && admit(queue[0], rows[queue[0]].label, prompts.get(queue[0])!)) {
      running.push(queue.shift()!);
      late++;
    }
  }
  snap(p.evicted.length);
  const before = p.evicted.length;
  const blank = p.freeCount() - p.lru.length;
  const other = Math.max(1, blank + Math.ceil(p.lru.length / 2));
  admit(k.requests, "another start", Array.from({ length: (other - 1) * poolBlock + poolOwn }, (_, i) => 50000 + i));
  snap(before);
  const later = p.evicted.length;
  finish(k.requests);
  admit(k.requests + 1, "the same start again", [...start, ...Array.from({ length: poolOwn }, (_, i) => 9000 + i)]);
  snap(later);
  return phases;
}
