import assert from "node:assert/strict";
import { test } from "node:test";
import { HEAD_DIM, PAIRS, angle, invFreq, rotate, score, seededVector, wavelength } from "../src/components/kernels/rope/rotation.ts";

const q = seededVector(7);
const k = seededVector(11);

function rotateHalfReference(x, pos) {
  const half = x.length / 2;
  const cos = Array.from({ length: x.length }, (_, d) => Math.cos(pos * invFreq(d % half)));
  const sin = Array.from({ length: x.length }, (_, d) => Math.sin(pos * invFreq(d % half)));
  const rotated = [...x.slice(half).map((v) => -v), ...x.slice(0, half)];
  return x.map((v, d) => v * cos[d] + rotated[d] * sin[d]);
}

test("the frequencies follow 10000^(-2i/64)", () => {
  assert.equal(invFreq(0), 1);
  assert.ok(Math.abs(invFreq(16) - 0.01) < 1e-12);
  assert.ok(Math.abs(wavelength(0) - 2 * Math.PI) < 1e-12);
  assert.ok(invFreq(PAIRS - 1) < invFreq(PAIRS - 2));
});

test("pairs are dimension i with dimension i + 32, as rotate_half does", () => {
  for (const pos of [0, 1, 37, 2048]) {
    const got = rotate(q, pos);
    const want = rotateHalfReference(q, pos);
    got.forEach((v, d) => assert.ok(Math.abs(v - want[d]) < 1e-9, `pos ${pos} dim ${d}`));
  }
});

test("position 0 is the identity and rotation keeps the length", () => {
  assert.deepEqual(rotate(q, 0), q);
  const norm = (v) => Math.hypot(...v);
  assert.ok(Math.abs(norm(rotate(q, 1234)) - norm(q)) < 1e-9);
});

test("the score depends only on the distance between the positions", () => {
  for (const [m, n, shift] of [[10, 3, 100], [2000, 1990, 48], [0, 0, 2048], [500, 900, 1000]]) {
    const base = score(q, m, k, n);
    assert.ok(Math.abs(score(q, m + shift, k, n + shift) - base) < 1e-9, `${m} ${n} +${shift}`);
  }
  assert.ok(Math.abs(score(q, 5, k, 5) - score(q, 0, k, 0)) < 1e-9);
  assert.ok(Math.abs(score(q, 12, k, 5) - score(q, 5, k, 12)) > 1e-3);
});

test("a fast pair turns many times where a slow one barely moves", () => {
  assert.ok(angle(0, 2048) > 2000);
  assert.ok(angle(HEAD_DIM / 2 - 1, 2048) < 0.3);
});
