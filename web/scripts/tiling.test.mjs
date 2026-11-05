import assert from "node:assert/strict";
import { test } from "node:test";
import { blockLoadBytes, tiling } from "../src/components/kernels/gemm/tiling.ts";

const ceil = { gbs: 300, tflops: 30 };
const gateUp = { out: 11264, in: 2048 };
const down = { out: 2048, in: 5632 };

test("naive reads a row of x and a row of W for every output", () => {
  const t = tiling({ out: 8, in: 4 }, 3, 16, 16, ceil);
  assert.equal(t.naiveBytes, 3 * 8 * 2 * 4 * 2);
});

test("with one row the weights are read once whatever the tile, so reuse stays under 2", () => {
  const small = tiling(gateUp, 1, 16, 32, ceil);
  const big = tiling(gateUp, 1, 128, 32, ceil);
  const weights = gateUp.out * gateUp.in * 2;
  for (const t of [small, big]) {
    assert.ok(t.loadBytes >= weights && t.loadBytes < weights * 1.07);
    assert.ok(t.reuse > 1.8 && t.reuse < 2);
  }
  assert.equal(small.bound, "memory");
  assert.ok(big.computeUs > small.computeUs * 7.9);
  assert.ok(big.memUs < small.memUs && big.memUs > small.memUs * 0.9);
  assert.ok(big.padWaste > 0.99);
  assert.equal(small.padWaste, 1 - 1 / 16);
});

test("with enough rows the tile edge is the reuse factor", () => {
  for (const tile of [16, 32, 64, 128]) {
    const t = tiling(gateUp, 512, tile, 32, ceil);
    assert.equal(t.reuse, tile);
    assert.equal(t.blocks, (512 / tile) * (gateUp.out / tile));
    assert.equal(t.padWaste, 0);
  }
});

test("512 rows of gate/up at tile 128", () => {
  const t = tiling(gateUp, 512, 128, 32, ceil);
  assert.equal(t.loadBytes, 2 * 88 * 512 * 2048 * 2);
  assert.equal(t.tileLoads, 4 * 88 * 64 * 2);
  assert.equal(t.sharedBytes, 2 * 2 * 128 * 32 * 2);
  assert.equal(t.floorBytes, (512 + 11264) * 2048 * 2);
});

test("shared memory grows with tile and k step, flops are fixed by the shape", () => {
  assert.equal(tiling(down, 16, 128, 128, ceil).sharedBytes, 131072);
  assert.equal(tiling(down, 16, 16, 16, ceil).sharedBytes, 2048);
  assert.equal(tiling(down, 64, 32, 32, ceil).flops, 2 * 64 * 2048 * 5632);
});

test("a large tile on many rows crosses to compute bound only when intensity passes the ridge", () => {
  const t = tiling(gateUp, 512, 128, 32, { gbs: 323.6, tflops: 34.5 });
  assert.equal(t.bound, "memory");
  assert.ok(t.intensity < t.ridge);
  const fast = tiling(gateUp, 512, 128, 32, { gbs: 323.6, tflops: 1 });
  assert.equal(fast.bound, "compute");
});

test("bytes loaded by one block grow with the k steps done and stop at K", () => {
  assert.equal(blockLoadBytes(512, 64, 32, 3, 2048), 128 * 96 * 2);
  assert.equal(blockLoadBytes(1, 64, 32, 3, 2048), 65 * 96 * 2);
  assert.equal(blockLoadBytes(512, 64, 128, 99, 2048), 128 * 2048 * 2);
});
