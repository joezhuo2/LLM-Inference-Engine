import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { number, parseLaunchGaps, parseProfiling, slug } from "./profiling.mjs";

const doc = `# Profiling

## Paged attention v2 (Branch 37)

Some text | with a pipe.

| rows | context | v1 (us) | v2 (us) |
| --- | --- | --- | --- |
| 1 | 2,048 | 600.2 | 165.0 |
| 1 | 8,192 | n/a | 582.7 |

More text.

### Small batches (Branch 33)

| tokens | fast |
| --- | --- |
| 1 | 3.8 |
`;

test("sections keep their title, level and branch", () => {
  const s = parseProfiling(doc);
  assert.deepEqual(
    s.map((x) => [x.id, x.level, x.branch]),
    [
      ["paged-attention-v2-branch-37", 2, 37],
      ["small-batches-branch-33", 3, 33],
    ],
  );
});

test("tables are read as header and rows of trimmed cells", () => {
  const [v2, small] = parseProfiling(doc);
  assert.equal(v2.tables.length, 1);
  assert.deepEqual(v2.tables[0].header, ["rows", "context", "v1 (us)", "v2 (us)"]);
  assert.deepEqual(v2.tables[0].rows[1], ["1", "8,192", "n/a", "582.7"]);
  assert.deepEqual(small.tables[0].rows, [["1", "3.8"]]);
});

test("a line with a pipe that is not followed by a separator is not a table", () => {
  assert.equal(parseProfiling("## A\n\n| just text |\nnext line\n")[0].tables.length, 0);
});

test("number reads thousands separators and leading values", () => {
  assert.equal(number("2,453.6"), 2453.6);
  assert.equal(number("11.8 (3.6 + 8.2)"), 11.8);
  assert.equal(number("n/a"), null);
  assert.equal(number("-0.5"), -0.5);
});

test("slug drops code ticks and punctuation", () => {
  assert.equal(slug("Fused `add_rmsnorm` (Branch 30)"), "fused-add-rmsnorm-branch-30");
});

test("the repository's profiling notes parse with the attention tables intact", () => {
  const sections = parseProfiling(readFileSync(new URL("../../docs/profiling.md", import.meta.url), "utf8"));
  const split = sections.find((s) => s.id === "split-context-attention-branch-40");
  assert.ok(split, "split section present");
  const table = split.tables[0];
  assert.deepEqual(table.header.slice(0, 3), ["rows", "context", "v2"]);
  assert.ok(table.rows.length >= 8);
  for (const row of table.rows) assert.equal(row.length, table.header.length);
});

test("launch gap lines are read per step kind", () => {
  const gaps = parseLaunchGaps(`## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 8069 us, median busy 6909 us, median idle 11.5% of the span, 17 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 11187 us, median busy 10093 us, median idle 8.1% of the span, 40 gaps over 5 us per step
- not a step line
`);
  assert.deepEqual(Object.keys(gaps), ["decode", "prefill"]);
  assert.deepEqual(gaps.decode, { steps: 252, kernels: 224, span_us: 8069, busy_us: 6909, idle_pct: 11.5, gaps: 17, gap_us: 5 });
});

test("the committed decode step profiles have a decode launch gap line", () => {
  for (const name of ["decode_step.md", "decode_step_graphs.md"]) {
    const gaps = parseLaunchGaps(readFileSync(new URL(`../../results/${name}`, import.meta.url), "utf8"));
    assert.ok(gaps.decode, `${name} has a decode line`);
    assert.ok(gaps.decode.busy_us <= gaps.decode.span_us);
  }
});
