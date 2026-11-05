import assert from "node:assert/strict";
import { test } from "node:test";
import { START_MAX, directAttention, makeProblem, maxDiff, pagedAttention, tokensIn } from "../src/components/kernels/attention/online-softmax.ts";

test("online softmax over blocks equals one-shot softmax attention", () => {
  for (const blockSize of [8, 16, 32]) {
    for (const contextLen of [1, 7, 16, 17, 100, 255, 256]) {
      for (let seed = 1; seed <= 5; seed++) {
        const p = makeProblem(contextLen, blockSize, seed);
        const { out } = pagedAttention(p);
        assert.ok(maxDiff(out, directAttention(p)) < 1e-5, `block ${blockSize} context ${contextLen} seed ${seed}`);
      }
    }
  }
});

test("the first block rescales an empty state by zero", () => {
  const p = makeProblem(40, 16, 3);
  const [first] = pagedAttention(p).steps;
  assert.equal(first.maxBefore, START_MAX);
  assert.equal(first.rescale, 0);
  assert.equal(first.sumBefore, 0);
  assert.ok(first.grew);
});

test("a block with a bigger score shrinks the old sum and accumulator by exp(old max minus new max)", () => {
  const p = makeProblem(200, 16, 11);
  const { steps } = pagedAttention(p);
  const raised = steps.slice(1).filter((s) => s.grew);
  assert.ok(raised.length > 0);
  for (const s of raised) {
    assert.ok(s.maxAfter > s.maxBefore && s.rescale < 1);
    assert.ok(Math.abs(s.rescale - Math.exp(s.maxBefore - s.maxAfter)) < 1e-6);
  }
  for (const s of steps.slice(1).filter((s) => !s.grew)) assert.equal(s.rescale, 1);
});

test("blocks sit scattered in the pool, never on block 0, and the last one may be partial", () => {
  const p = makeProblem(100, 32, 5);
  assert.equal(p.table.length, 4);
  assert.equal(new Set(p.table).size, 4);
  assert.ok(p.table.every((b) => b >= 1 && b < p.poolBlocks));
  assert.deepEqual([0, 1, 2, 3].map((i) => tokensIn(p, i)), [32, 32, 32, 4]);
  assert.equal(pagedAttention(p).steps.reduce((n, s) => n + s.tokens, 0), 100);
});

test("the same seed gives the same problem", () => {
  assert.deepEqual(pagedAttention(makeProblem(64, 16, 9)).out, pagedAttention(makeProblem(64, 16, 9)).out);
});
