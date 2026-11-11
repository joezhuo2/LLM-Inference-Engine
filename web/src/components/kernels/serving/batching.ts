export type Grid = { slots: number; steps: number; cells: (number | null)[][]; redo: boolean[][]; preempted: number[][]; running: number[]; waiting: number[][]; evictions: number[]; wasted: number };

export const lengths = [3, 9, 2, 5, 4, 2, 6, 3, 2, 4];
export const slots = 4;
const blockSteps = 2;

const blocksFor = (n: number) => Math.ceil(n / blockSteps);

function empty(steps: number): Pick<Grid, "cells" | "redo" | "preempted"> {
  return {
    cells: Array.from({ length: slots }, () => Array<number | null>(steps).fill(null)),
    redo: Array.from({ length: slots }, () => Array<boolean>(steps).fill(false)),
    preempted: Array.from({ length: slots }, () => []),
  };
}

function finish(g: Omit<Grid, "wasted" | "slots">): Grid {
  const used = lengths.reduce((n, l) => n + l, 0);
  const written = g.cells.flatMap((row, s) => row.filter((id, t) => id !== null && !g.redo[s][t])).length;
  if (written !== used) throw new Error(`batching grid wrote ${written} cells for ${used} steps of work`);
  return { ...g, slots, wasted: slots * g.steps - used };
}

export function staticBatching(): Grid {
  const rows: number[][] = [];
  for (let i = 0; i < lengths.length; i += slots) rows.push(lengths.slice(i, i + slots).map((_, j) => i + j));
  const total = rows.reduce((n, ids) => n + Math.max(...ids.map((id) => lengths[id])), 0);
  const g = empty(total);
  const running: number[] = [];
  const waiting: number[][] = [];
  let t0 = 0;
  for (const ids of rows) {
    const span = Math.max(...ids.map((id) => lengths[id]));
    for (let t = 0; t < span; t++) {
      ids.forEach((id, s) => {
        if (t < lengths[id]) g.cells[s][t0 + t] = id;
      });
      running.push(ids.filter((id) => t < lengths[id]).length);
      waiting.push(lengths.map((_, id) => id).filter((id) => id >= ids[ids.length - 1] + 1));
    }
    t0 += span;
  }
  return finish({ ...g, steps: total, running, waiting, evictions: Array(total).fill(0) });
}

export function continuousBatching(poolBlocks: number): Grid {
  const done = lengths.map(() => 0);
  const held = lengths.map(() => 0);
  const slotOf = lengths.map(() => -1);
  const seen = lengths.map(() => 0);
  const queue = lengths.map((_, id) => id);
  let run: number[] = [];
  let free = poolBlocks;
  const cols: { id: number | null; redo: boolean }[][] = [];
  const out = { running: [] as number[], waiting: [] as number[][], evictions: [] as number[] };
  const gone: { slot: number; step: number }[] = [];
  const release = (id: number) => {
    free += held[id];
    held[id] = 0;
    slotOf[id] = -1;
  };
  for (let t = 0; run.length || queue.length; t++) {
    const fresh = new Set<number>();
    while (queue.length && run.length < slots) {
      const id = queue[0];
      const need = blocksFor(done[id] + 1);
      if (need > free) break;
      queue.shift();
      free -= need;
      held[id] = need;
      slotOf[id] = [...Array(slots).keys()].find((s) => !run.some((r) => slotOf[r] === s))!;
      run.push(id);
      fresh.add(id);
    }
    let evicted = 0;
    for (let i = 0; i < run.length; ) {
      const id = run[i];
      let dropped = false;
      while (!fresh.has(id) && blocksFor(done[id] + 1) > held[id] && free === 0) {
        const victim = run.pop()!;
        gone.push({ slot: slotOf[victim], step: t });
        release(victim);
        done[victim] = 0;
        queue.unshift(victim);
        evicted++;
        if (victim === id) {
          dropped = true;
          break;
        }
      }
      if (dropped) continue;
      const need = blocksFor(done[id] + 1);
      free -= need - held[id];
      held[id] = need;
      i++;
    }
    const col: { id: number | null; redo: boolean }[] = Array.from({ length: slots }, () => ({ id: null, redo: false }));
    for (const id of run) col[slotOf[id]] = { id, redo: seen[id] > 0 && done[id] < seen[id] };
    for (const id of run) {
      done[id]++;
      seen[id] = Math.max(seen[id], done[id]);
    }
    out.running.push(run.length);
    out.waiting.push([...queue]);
    out.evictions.push(evicted);
    cols.push(col);
    for (const id of run.filter((r) => done[r] === lengths[r])) release(id);
    run = run.filter((r) => done[r] < lengths[r]);
  }
  const g = empty(cols.length);
  cols.forEach((col, t) =>
    col.forEach((c, s) => {
      g.cells[s][t] = c.id;
      g.redo[s][t] = c.redo;
    }),
  );
  gone.forEach(({ slot, step }) => g.preempted[slot].push(step));
  return finish({ ...g, steps: cols.length, ...out });
}
