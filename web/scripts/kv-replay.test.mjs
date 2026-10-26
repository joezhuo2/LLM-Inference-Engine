import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { replayKv } from "./kv-replay.mjs";

const meta = { kv_blocks: 8, block_size: 4 };
const requests = [
  { id: 0, prompt_tokens: 5, output_tokens: 4, arrival_s: 0, preemptions: 0 },
  { id: 1, prompt_tokens: 3, output_tokens: 2, arrival_s: 0.1, preemptions: 0 },
  { id: 2, prompt_tokens: 4, output_tokens: 1, arrival_s: 0.2, preemptions: 0 },
];
const step = (index, kind, seqs, finished, free_blocks) => ({ index, start_s: index / 10, end_s: (index + 1) / 10, kind, seqs, finished, free_blocks });
const steps = [step(0, "prefill", [0, 1], [], 4), step(1, "decode", [0, 1], [], 4), step(2, "decode", [0, 1], [1], 5), step(3, "prefill", [2], [], 4), step(4, "decode", [0, 2], [0, 2], 7)];

test("blocks follow first fit and come back when a sequence finishes", () => {
  const r = replayKv({ meta, requests, steps }, "t.json");
  assert.deepEqual(
    r.requests.map((q) => [q.id, q.blocks.map((b) => [b.id, b.step]), q.admitted, q.finished, q.final_length]),
    [
      [0, [[1, 0], [2, 0]], 0, 4, 8],
      [1, [[3, 0], [4, 2]], 0, 2, 5],
      [2, [[3, 3], [4, 4]], 3, 4, 5],
    ],
  );
  assert.deepEqual(r.steps.map((s) => s[2]), [3, 3, 2, 3, 0]);
  assert.deepEqual(r.steps.map((s) => s[3]), [0, 1, 2, 2, 3]);
  assert.deepEqual([r.peak.blocks, r.peak.step, r.peak.running, r.peak.tokens, r.max_block, r.max_running], [3, 0, 2, 8, 4, 2]);
});

test("a free count that disagrees with the trace fails", () => {
  const bad = steps.map((s) => (s.index === 1 ? { ...s, free_blocks: 3 } : s));
  assert.throws(() => replayKv({ meta, requests, steps: bad }, "t.json"), /step 1 leaves 4 blocks free, the trace says 3/);
});

test("a trace with preemptions is refused", () => {
  assert.throws(() => replayKv({ meta, requests: requests.map((q) => ({ ...q, preemptions: 1 })), steps }, "t.json"), /preemption/);
});

test("the recorded chat trace replays step for step", () => {
  const path = new URL("../../results/traces/chat_b16_rate4.json", import.meta.url);
  if (!existsSync(path)) return;
  const r = replayKv(JSON.parse(readFileSync(path, "utf8")), "chat_b16_rate4.json");
  assert.equal(r.steps.length, 751);
  assert.equal(r.requests.length, 24);
  assert.ok(r.peak.blocks > 0 && r.max_block < 128);
});
