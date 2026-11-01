import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { DashModel, UNKNOWN } from "../src/lib/dashboard/model.ts";
import { percentile, replayTiles, spread, stepsDoneBy } from "../src/lib/dashboard/stats.ts";
import { replayKv } from "./kv-replay.mjs";

const path = new URL("../../results/traces/chat_b16_rate4.json", import.meta.url);
const trace = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;

function model(t, withPrompts) {
  const prompts = withPrompts ? new Map(t.requests.map((r) => [r.id, r.prompt_tokens])) : undefined;
  return new DashModel({ totalBlocks: t.meta.kv_blocks, blockSize: t.meta.block_size, slots: 8, prompts });
}

test("with prompt lengths the model hands out the same blocks as the build-time replay", { skip: !trace }, () => {
  const m = model(trace, true);
  const held = new Map();
  for (const s of trace.steps) {
    const table = (seq) => seq.blocks.map((b, j) => [b, seq.since[j]]);
    m.apply(s);
    for (const seq of m.seqs.values()) if (seq.running) held.set(seq.id, table(seq));
    assert.equal(m.free(), s.free_blocks, `step ${s.index}`);
    assert.equal(m.unknown, 0);
  }
  const kv = replayKv(trace, "t.json");
  for (const r of kv.requests) assert.deepEqual(held.get(r.id), r.blocks.map((b) => [b.id, b.step]), `request ${r.id}`);
  assert.equal(m.missed, 0);
  assert.equal(m.highWater, kv.max_block);
});

test("every request keeps one color slot while it runs and slots are reused", { skip: !trace }, () => {
  const m = model(trace, true);
  for (const s of trace.steps) {
    m.apply(s);
    const slots = m.liveSeqs().map((q) => q.slot);
    assert.equal(new Set(slots).size, slots.length);
    assert.ok(slots.every((k) => k >= 0 && k < 8));
  }
  assert.equal(m.seqs.size, 24);
  assert.ok([...m.seqs.values()].every((q) => q.finished >= 0 && q.text.length > 0));
});

test("without prompt lengths the totals still match the server and most blocks are attributed", { skip: !trace }, () => {
  const exact = model(trace, true);
  const guess = model(trace, false);
  let seen = 0;
  let wrong = 0;
  for (const s of trace.steps) {
    exact.apply(s);
    guess.apply(s);
    assert.equal(guess.free(), s.free_blocks, `step ${s.index}`);
    assert.ok(guess.unknown <= 1, `step ${s.index}: ${guess.unknown} unknown blocks`);
    for (const q of exact.liveSeqs()) {
      seen++;
      if (q.blocks.length !== guess.seqs.get(q.id).blocks.length) wrong++;
    }
  }
  assert.ok(wrong / seen < 0.12, `${wrong} of ${seen} block counts differ`);
});

test("a gap in the step indices is counted and the pool is reconciled with unknown blocks", { skip: !trace }, () => {
  const m = model(trace, false);
  for (const s of trace.steps.slice(0, 100)) m.apply(s);
  for (const s of trace.steps.slice(130)) {
    m.apply(s);
    assert.equal(m.free(), s.free_blocks, `step ${s.index}`);
  }
  assert.equal(m.missed, 30);
  assert.equal(m.gaps.get(130), 30);
});

test("joining a server mid-run puts the blocks of earlier requests in the unknown pool", () => {
  const m = new DashModel({ totalBlocks: 64, blockSize: 4, slots: 8 });
  const step = (index, kind, seqs, finished, free_blocks) => ({ index, start_s: index, end_s: index + 0.5, kind, seqs, tokens: seqs.map(() => ({ id: 1, text: "a" })), finished, waiting: 0, running: seqs.length - finished.length, free_blocks });
  m.apply(step(40, "decode", [7, 8], [], 50));
  assert.equal(m.unknown, 13);
  assert.equal([...m.owner].filter((o) => o === UNKNOWN).length, 13);
  m.apply(step(41, "prefill", [9], [], 47));
  assert.equal(m.seqs.get(9).blocks.length, 3);
  m.apply(step(42, "decode", [7, 8, 9], [7], 52));
  assert.equal(m.free(), 52);
  assert.equal(m.unknown, 8);
  assert.deepEqual(m.liveSeqs().map((q) => q.id), [8, 9]);
});

test("tiles read the recorded run at a point in time", { skip: !trace }, () => {
  const done = stepsDoneBy(trace.steps, 3);
  const tiles = replayTiles(trace.steps, trace.requests, done, 3, trace.meta.kv_blocks);
  const last = trace.steps[done - 1];
  assert.equal(tiles.running, last.running);
  assert.equal(tiles.kvUsed, trace.meta.kv_blocks - 1 - last.free_blocks);
  const finished = trace.requests.filter((r) => r.finished_s <= 3);
  assert.ok(finished.length > 0 && tiles.ttft && tiles.ttft.p50 <= tiles.ttft.p99);
  assert.ok(tiles.tokensPerS > 100);
});

test("percentiles use the nearest rank", () => {
  assert.deepEqual(spread([5, 1, 3, 2, 4]), { p50: 3, p99: 5 });
  assert.equal(spread([]), null);
  assert.equal(percentile([1, 2, 3, 4], 50), 2);
});
