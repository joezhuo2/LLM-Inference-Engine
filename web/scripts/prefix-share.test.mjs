import assert from "node:assert/strict";
import { test } from "node:test";
import { cachedOf, shareRun } from "../src/components/kernels/prefix-caching/share.ts";

const knobs = (shared, atOnce) => ({ shared, own: 64, requests: 32, atOnce, budget: 2048, maxBatch: 16, block: 16 });

test("a hit reuses only the whole blocks of the shared prefix, and never the block of the last prompt token", () => {
  assert.equal(cachedOf(1024, 64, 16), 1024);
  assert.equal(cachedOf(1000, 64, 16), 992);
  assert.equal(cachedOf(15, 64, 16), 0);
  assert.equal(cachedOf(32, 0, 16), 16);
  assert.equal(cachedOf(32, 1, 16), 32);
});

test("one request at a time, every request after the first finds the whole prefix", () => {
  const run = shareRun(knobs(1024, false));
  assert.equal(run.firstStep, 1);
  assert.equal(run.cachedTotal, 31 * 1024);
  assert.equal(run.hitRate.toFixed(4), ((31 * 1024) / (32 * 1088)).toFixed(4));
});

test("all at once, the first step's prompts fit the token budget and find nothing", () => {
  assert.deepEqual([256, 512, 1024].map((p) => shareRun(knobs(p, true)).firstStep), [6, 3, 1]);
  assert.equal((100 * shareRun(knobs(256, true)).hitRate).toFixed(1), "65.0");
  assert.equal((100 * shareRun(knobs(512, true)).hitRate).toFixed(1), "80.6");
  assert.equal(shareRun({ ...knobs(16, true), maxBatch: 4 }).firstStep, 4);
});
