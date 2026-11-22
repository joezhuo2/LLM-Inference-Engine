import assert from "node:assert/strict";
import { test } from "node:test";
import { MiniPool, poolRun } from "../src/components/kernels/prefix-caching/pool.ts";

test("requests with the same start share its blocks, each held by all of them, and take the lowest blank block for the rest", () => {
  const [arrive] = poolRun({ shared: 3, requests: 3, pool: 10 });
  assert.deepEqual(arrive.rows.map((r) => r.blocks), [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6]]);
  assert.deepEqual(arrive.slots.slice(0, 6).map((s) => s.count), [3, 3, 3, 1, 1, 1]);
  assert.equal(arrive.reused, 2 * 48);
  assert.equal(arrive.prefilled, 56 + 8 + 8);
  assert.equal(arrive.held, 6);
});

test("released published blocks stay cached; another start takes blank blocks first, then evicts the end of the chain first", () => {
  const [, done, other, again] = poolRun({ shared: 3, requests: 3, pool: 10 });
  assert.deepEqual(done.slots.map((s) => s.state).slice(0, 4), ["cached", "cached", "cached", "blank"]);
  assert.equal(done.cached, 3);
  assert.equal(done.blank, 7);
  assert.deepEqual(other.evicted, [3, 2]);
  assert.deepEqual(other.rows[3].blocks, [4, 5, 6, 7, 8, 9, 10, 3, 2]);
  assert.equal(again.rows[4].reused, 16);
  assert.deepEqual(again.rows[4].blocks, [1, 2, 3, 10]);
  assert.deepEqual(again.evicted, [3, 10]);
});

test("a request waits when the pool has no room even with sharing, and sharing fits requests that would not fit alone", () => {
  const [arrive] = poolRun({ shared: 2, requests: 5, pool: 6 });
  assert.deepEqual(arrive.rows.map((r) => r.state), ["running", "running", "running", "running", "waiting"]);
  assert.equal(arrive.rows[4].need, 1);
});

test("the block of the last token is never matched, and a failed check changes nothing", () => {
  const p = new MiniPool(4, 4);
  p.allocateTokens(0, [1, 2, 3, 4, 5, 6, 7, 8]);
  p.publish(0, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(p.matchBlocks([1, 2, 3, 4, 5, 6, 7, 8]), [1]);
  assert.equal(p.canAllocateTokens([9, 9, 9, 9, 9, 9, 9, 9, 9]), false);
  assert.throws(() => p.allocateTokens(1, [9, 9, 9, 9, 9, 9, 9, 9, 9]), /no room/);
  assert.equal(p.freeCount(), 1);
});
