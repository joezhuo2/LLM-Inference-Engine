import assert from "node:assert/strict";
import { test } from "node:test";
import { median, prefixRows, prefixRun } from "./prefix-runs.mjs";

const summary = (cache, shared, extra = {}) => ({ prefix_cache: cache === "on" ? 1 : 0, shared_prefix: shared, prompt_len: 64, prefix_hit_rate: 0.5, output_throughput: 100, prefill_tokens: 1000, cached_tokens: 500, date: "2026-11-22T00:00:00Z", gpu: "GPU", ...extra });
const requests = (ttfts) => ttfts.map((ttft_s, id) => ({ id, ttft_s }));

test("a run takes request 0 as the cold first request and the median of the others as the later ones", () => {
  const r = prefixRun("on_p1024_q2_r1_summary.csv", summary("on", 1024), requests([0.128, 0.03, 0.02, 0.025]));
  assert.equal(r.cache, "on");
  assert.equal(r.shared, 1024);
  assert.equal(r.rate, 2);
  assert.equal(r.run, 1);
  assert.equal(r.first_ttft_ms, 128);
  assert.equal(r.later_ttft_ms, 25);
});

test("a file name or a summary that does not say what the run was is refused", () => {
  assert.throws(() => prefixRun("on_p1024_r1_summary.csv", summary("on", 1024), requests([1, 1])), /name must read/);
  assert.throws(() => prefixRun("on_p512_q0_r1_summary.csv", summary("on", 1024), requests([1, 1])), /does not match the file name/);
  assert.throws(() => prefixRun("off_p512_q0_r1_summary.csv", summary("on", 512), requests([1, 1])), /does not match the file name/);
  assert.throws(() => prefixRun("on_p512_q0_r1_summary.csv", summary("on", 512), requests([1])), /later request/);
});

test("rows are medians over the runs of one setting, in the order of the results.md table", () => {
  const runs = [
    prefixRun("off_p256_q0_r1_summary.csv", summary("off", 256, { output_throughput: 480 }), requests([1, 2])),
    prefixRun("on_p256_q0_r1_summary.csv", summary("on", 256, { output_throughput: 700 }), requests([1, 2])),
    prefixRun("on_p256_q0_r2_summary.csv", summary("on", 256, { output_throughput: 781 }), requests([1, 2])),
    prefixRun("on_p256_q0_r3_summary.csv", summary("on", 256, { output_throughput: 790 }), requests([1, 2])),
    prefixRun("on_p256_q2_r1_summary.csv", summary("on", 256), requests([1, 2])),
  ];
  const rows = prefixRows(runs);
  assert.deepEqual(rows.map((r) => `${r.arrivals} ${r.cache} ${r.runs}`), ["2/s on 1", "at once on 3", "at once off 1"]);
  assert.equal(rows[1].output_tps, 781);
  assert.equal(median([4, 1, 3, 2]), 2.5);
});
