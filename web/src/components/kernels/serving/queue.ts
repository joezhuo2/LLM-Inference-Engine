import { Scheduler } from "../chunked-prefill/scheduler-model";

export const queueConfig = { numBlocks: 13, blockSize: 4, maxBatch: 4, maxPrefillTokens: 32, maxContext: 32 };
export const queueBlocks = queueConfig.numBlocks - 1;

export type QueueRequest = { name: string; prompt: number; maxNew: number };

export const queueRequests: QueueRequest[] = [
  { name: "A", prompt: 6, maxNew: 10 },
  { name: "B", prompt: 4, maxNew: 6 },
  { name: "C", prompt: 7, maxNew: 9 },
  { name: "D", prompt: 10, maxNew: 8 },
  { name: "E", prompt: 18, maxNew: 4 },
  { name: "F", prompt: 3, maxNew: 5 },
];

export type Wait = { id: number; reason: "batch" | "blocks" | "next"; need: number };

export type Snapshot = {
  step: number;
  kind: "start" | "prefill" | "decode";
  seqs: number[];
  admitted: number[];
  evicted: number[];
  finished: number[];
  needy: number | null;
  grew: number[];
  waiting: number[];
  running: number[];
  done: number[];
  held: number[];
  cached: number[];
  output: number[];
  owner: number[];
  free: number;
  wait: Wait | null;
  note: string;
};

const names = (ids: number[]) => {
  const list = ids.map((i) => queueRequests[i].name);
  return list.length < 2 ? list.join("") : `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`;
};
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

function simulate(): Snapshot[] {
  const s = new Scheduler(queueConfig);
  queueRequests.forEach((r, id) => s.submit(id, r.prompt, r.maxNew));
  const owner: number[] = Array(queueConfig.numBlocks).fill(-1);
  const tables = new Map<number, number[]>();
  const sync = () => {
    for (const [id, blocks] of [...tables]) {
      if (!s.blocks.tables.has(id)) {
        for (const b of blocks) owner[b] = -1;
        tables.delete(id);
      }
    }
    for (const id of s.running) {
      const want = s.blocks.tables.get(id) ?? 0;
      const mine = tables.get(id) ?? [];
      while (mine.length < want) {
        const b = owner.indexOf(-1, 1);
        owner[b] = id;
        mine.push(b);
      }
      tables.set(id, mine);
    }
  };
  const done: number[] = [];
  const state = () => {
    const ids = queueRequests.map((_, id) => id);
    const front = s.waiting[0];
    let wait: Wait | null = null;
    if (front !== undefined) {
      const r = s.requests.get(front)!;
      const need = Math.ceil((r.prompt + r.output) / queueConfig.blockSize);
      wait = { id: front, need, reason: s.running.length >= queueConfig.maxBatch ? "batch" : need > s.blocks.free ? "blocks" : "next" };
    }
    return {
      waiting: [...s.waiting],
      running: [...s.running],
      done: [...done],
      held: ids.map((id) => s.blocks.tables.get(id) ?? 0),
      cached: ids.map((id) => s.blocks.lengths.get(id) ?? 0),
      output: ids.map((id) => s.requests.get(id)!.output),
      owner: [...owner],
      free: s.blocks.free,
      wait,
    };
  };
  const first = state();
  const out: Snapshot[] = [{ step: 0, kind: "start", seqs: [], admitted: [], evicted: [], finished: [], needy: null, grew: [], ...first, note: `${plural(queueRequests.length, "request")} wait in arrival order; the pool has ${queueBlocks} free blocks of ${queueConfig.blockSize} tokens and max_batch is ${queueConfig.maxBatch}` }];
  for (let t = 1; t < 80 && (s.waiting.length || s.running.length); t++) {
    const before = { held: new Map(s.blocks.tables), preemptions: new Map([...s.requests].map(([id, r]) => [id, r.preemptions])), running: [...s.running], free: s.blocks.free, lengths: new Map(s.blocks.lengths) };
    const step = s.schedule();
    const evicted = [...s.requests].filter(([id, r]) => r.preemptions > (before.preemptions.get(id) ?? 0)).map(([id]) => id);
    let needy: number | null = null;
    if (!step.prefill && evicted.length) {
      let free = before.free;
      for (const id of before.running) {
        if ((before.lengths.get(id) ?? 1) % queueConfig.blockSize !== 0) continue;
        if (free > 0) free--;
        else {
          needy = id;
          break;
        }
      }
    }
    const grew = step.prefill ? [] : step.seqs.filter((id) => (s.blocks.tables.get(id) ?? 0) > (before.held.get(id) ?? 0));
    const taken = step.seqs.reduce((n, id) => n + (s.blocks.tables.get(id) ?? 0), 0);
    sync();
    if (step.seqs.length) s.update();
    const finished = step.seqs.filter((id) => s.requests.get(id)!.finished);
    done.push(...finished);
    sync();
    const now = state();
    const parts: string[] = [];
    if (step.prefill) {
      const again = step.seqs.filter((id) => (before.preemptions.get(id) ?? 0) > 0);
      parts.push(`${names(step.seqs)} ${step.seqs.length === 1 ? "is" : "are"} admitted and prefilled, ${plural(taken, "block")} taken`);
      if (again.length) parts.push(`${names(again)} recomputes its prompt and the tokens it had written, from position 0`);
    } else {
      if (evicted.length && needy !== null) parts.push(evicted.includes(needy) ? `${names([needy])} needs a block, none is free, and it is the back of the running list itself: it loses its ${plural(before.held.get(needy) ?? 0, "block")} and goes to the front of the queue` : `${names([needy])} needs a block and none is free: ${names(evicted)}, the back of the running list, loses its ${plural(evicted.reduce((n, id) => n + (before.held.get(id) ?? 0), 0), "block")} and goes to the front of the queue`);
      parts.push(`${names(step.seqs)} ${step.seqs.length === 1 ? "decodes" : "decode"} one token`);
      const quiet = grew.filter((id) => id !== needy);
      if (quiet.length) parts.push(`${names(quiet)} ${quiet.length === 1 ? "starts" : "start"} a new block`);
    }
    if (finished.length) parts.push(`${names(finished)} ${finished.length === 1 ? "finishes and frees its blocks" : "finish and free their blocks"}`);
    if (now.wait && now.wait.reason === "batch") parts.push(`${names([now.wait.id])} waits: all ${queueConfig.maxBatch} slots are taken`);
    if (now.wait && now.wait.reason === "blocks") parts.push(`${names([now.wait.id])} needs ${plural(now.wait.need, "block")} and ${now.free} ${now.free === 1 ? "is" : "are"} free, so it waits${now.waiting.length > 1 ? ", and nothing behind it may skip ahead" : ""}`);
    out.push({ step: t, kind: step.prefill ? "prefill" : "decode", seqs: step.seqs, admitted: step.prefill ? step.seqs : [], evicted, finished, needy, grew, ...now, note: parts.join("; ") });
  }
  return out;
}

export const queueRun = simulate();
export const queueEvictions = queueRun.reduce((n, s) => n + s.evicted.length, 0);
export const firstEviction = queueRun.findIndex((s) => s.evicted.length > 0);
export const firstBlocked = queueRun.findIndex((s) => s.wait?.reason === "blocks" && s.waiting.length > 1);
export const selfEviction = queueRun.findIndex((s) => s.needy !== null && s.evicted.includes(s.needy));
