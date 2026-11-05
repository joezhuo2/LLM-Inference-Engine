import assert from "node:assert/strict";
import { test } from "node:test";
import { buildGroups, draw, stages, topKKeep, topPKeep } from "../src/components/kernels/sampling/sampler.ts";

const dump = { tokens: ["a", "b", "c", "d"], logits: [3, 2, 1, 1], tail: [[0.5, 10], [0, 20]] };
const groups = buildGroups(dump);
const off = { temperature: 1, topK: 0, topP: 1 };
const sum = (v) => v.reduce((a, b) => a + b, 0);

test("softmax over groups sums to one and counts weigh the tail", () => {
  const s = stages(groups, off);
  assert.ok(Math.abs(sum(s.base) - 1) < 1e-12);
  assert.ok(Math.abs(s.base[4] - (10 * Math.exp(0.5)) / sum(groups.map((g) => g.count * Math.exp(g.logit)))) < 1e-12);
  assert.equal(s.survivors, 34);
});

test("top-k keeps whole groups until k tokens are counted, ties included", () => {
  assert.deepEqual(topKKeep([1, 1, 1, 1, 10], 3), [true, true, true, false, false]);
  assert.deepEqual(topKKeep([1, 1, 1, 1, 10], 0), [true, true, true, true, true]);
  assert.equal(stages(groups, { ...off, topK: 2 }).survivors, 2);
  assert.equal(stages(groups, { ...off, topK: 3 }).survivors, 3);
});

test("top-p keeps the smallest prefix whose mass reaches p", () => {
  const keep = [true, true, true, true];
  assert.deepEqual(topPKeep([0.5, 0.3, 0.15, 0.05], keep, 0.8), [true, true, false, false]);
  assert.deepEqual(topPKeep([0.5, 0.3, 0.15, 0.05], keep, 0.81), [true, true, true, false]);
  assert.deepEqual(topPKeep([0.5, 0.3, 0.15, 0.05], keep, 1), keep);
});

test("top-p runs on the top-k survivors", () => {
  const s = stages(groups, { temperature: 1, topK: 2, topP: 0.5 });
  assert.equal(s.survivors, 1);
  assert.ok(Math.abs(s.final[0] - 1) < 1e-12);
});

test("raising temperature never shrinks the entropy, zero is greedy", () => {
  let last = -1;
  for (const t of [0.25, 0.5, 1, 2, 4]) {
    const h = stages(groups, { ...off, temperature: t }).entropy;
    assert.ok(h > last);
    last = h;
  }
  const g = stages(groups, { ...off, temperature: 0, topK: 2 });
  assert.equal(g.survivors, 1);
  assert.equal(g.entropy, 0);
  assert.equal(draw(groups, { temperature: 0, topK: 2, topP: 0.1 }, 5, 9), 0);
});

test("the same seed and step repeat, and Gumbel-max draws follow the distribution", () => {
  assert.equal(draw(groups, off, 7, 3), draw(groups, off, 7, 3));
  const knobs = { temperature: 1.3, topK: 3, topP: 1 };
  const s = stages(groups, knobs);
  const n = 40000;
  const hits = groups.map(() => 0);
  for (let step = 0; step < n; step++) hits[draw(groups, knobs, 11, step)]++;
  groups.forEach((_, i) => assert.ok(Math.abs(hits[i] / n - s.final[i]) < 0.01, `group ${i}: ${hits[i] / n} against ${s.final[i]}`));
  assert.equal(hits[3] + hits[4] + hits[5], 0);
});

test("a tail group draws as often as its summed mass says", () => {
  const s = stages(groups, off);
  const n = 40000;
  const tail = groups.map((_, i) => i).filter((i) => groups[i].label === "other");
  let hits = 0;
  for (let step = 0; step < n; step++) hits += tail.includes(draw(groups, off, 3, step)) ? 1 : 0;
  assert.ok(Math.abs(hits / n - sum(tail.map((i) => s.final[i]))) < 0.01);
});
