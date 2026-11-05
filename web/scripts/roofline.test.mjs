import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { analyse, crossoverRows, opCost, rowSteps, rowsAt, stepOf } from "../src/components/kernels/foundations/roofline.ts";
import { parseModelShape, parseRooflineContext } from "./model-shape.mjs";
import { parseProfiling } from "./profiling.mjs";

const results = readFileSync(new URL("../../docs/results.md", import.meta.url), "utf8");
const shape = parseModelShape(readFileSync(new URL("../../tests/unit/test_config.cpp", import.meta.url), "utf8"));
const context = parseRooflineContext(results);
const ceilings = { gbs: 323.6, tflops: 34.5 };
const table = parseProfiling(results).find((s) => s.id === "roofline").tables[0];
const col = (name) => table.header.indexOf(name);

test("FLOP per byte matches the roofline table of results.md for every measured op", () => {
  for (const row of table.rows) {
    const [op, rows] = [row[col("op")], Number(row[col("rows")])];
    if (op === "lm_head") continue;
    const cost = opCost(shape, op, rows, context);
    assert.equal((cost.flops / cost.bytes).toFixed(1), Number(row[col("FLOP/byte")]).toFixed(1), `${op} at ${rows} rows`);
  }
});

test("time at the memory ceiling is within 10% of the measured GEMV time at one row", () => {
  for (const op of ["qkv", "o_proj", "gate_up", "down"]) {
    const measured = Number(table.rows.find((r) => r[col("op")] === op && r[col("rows")] === "1")[col("us/step")]);
    const predicted = analyse(opCost(shape, op, 1, context), ceilings).timeUs;
    assert.ok(Math.abs(predicted / measured - 1) < 0.1, `${op}: ${predicted} against ${measured}`);
  }
});

test("rows follow a log scale from 1 to 512 and every step changes them", () => {
  const rows = Array.from({ length: rowSteps + 1 }, (_, i) => rowsAt(i));
  assert.equal(rows[0], 1);
  assert.equal(rows.at(-1), 512);
  assert.ok(rows.every((r, i) => i === 0 || r > rows[i - 1]));
  assert.deepEqual([1, 16, 64, 512].map((r) => rowsAt(stepOf(r))), [1, 16, 64, 512]);
});

test("a GEMM crosses the ridge between 100 and 130 rows and is memory bound before", () => {
  for (const op of ["qkv", "o_proj", "gate_up", "down"]) {
    const at = crossoverRows(shape, op, context, ceilings);
    assert.ok(at > 100 && at < 130, `${op} crosses at ${at}`);
    assert.equal(analyse(opCost(shape, op, at - 1, context), ceilings).bound, "memory");
    assert.equal(analyse(opCost(shape, op, at, context), ceilings).bound, "compute");
  }
});

test("ops without a big weight matrix never cross, and attention, swiglu and rope do not depend on rows", () => {
  for (const op of ["attention", "rmsnorm", "swiglu", "rope"]) {
    assert.equal(crossoverRows(shape, op, 2048, ceilings), null, op);
    const b = opCost(shape, op, 512, 2048);
    assert.ok(b.flops / b.bytes < 10, op);
  }
  for (const op of ["attention", "swiglu", "rope"]) {
    const a = opCost(shape, op, 1, 512);
    const b = opCost(shape, op, 64, 512);
    assert.ok(Math.abs(a.flops / a.bytes / (b.flops / b.bytes) - 1) < 0.01, op);
  }
});

test("attention intensity grows with context toward the grouped-query ratio", () => {
  const at = (ctx) => {
    const c = opCost(shape, "attention", 1, ctx);
    return c.flops / c.bytes;
  };
  assert.ok(at(16) < at(96) && at(96) < at(2048) && at(2048) < shape.heads / shape.kv_heads);
});
