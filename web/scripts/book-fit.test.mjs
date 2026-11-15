import assert from "node:assert/strict";
import { test } from "node:test";
import { listRows } from "../src/components/book/fit.ts";

test("a long list fills the space to its bottom and keeps titles near the page number size", () => {
  const { rows, size } = listRows(20, [0, 11, 13, 16, 17], 380, 1316, 92);
  const entries = rows.filter((r) => r.k !== undefined);
  assert.equal(entries.length, 20);
  assert.ok(Math.abs(rows.at(-1).y - 1316) < 1e-9);
  assert.ok(size >= 28 && size <= 46);
  assert.ok(rows.every((r, i) => i === 0 || r.y > rows[i - 1].y));
});

test("a short list keeps the full pitch and type", () => {
  const { rows, step, size } = listRows(4, [], 380, 1316, 92);
  assert.equal(step, 92);
  assert.equal(size, 46);
  assert.ok(rows.at(-1).y < 1316);
});
