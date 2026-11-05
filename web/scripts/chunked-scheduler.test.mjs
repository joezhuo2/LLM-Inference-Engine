import assert from "node:assert/strict";
import { test } from "node:test";
import { isEmpty, numTokens, Playground, predict, Scheduler } from "../src/components/kernels/chunked-prefill/scheduler-model.ts";

const config = (numBlocks, maxBatch, maxContext = 2048) => ({ numBlocks, blockSize: 16, maxBatch, maxPrefillTokens: maxContext, maxContext });

function run(s, budget) {
  const step = s.scheduleChunked(budget);
  if (!isEmpty(step)) s.update();
  return step;
}

const chunks = (step) => step.prefill.map((c) => [c.id, c.first, c.count]);

test("a budget below max_batch is refused and changes nothing", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 8, 2);
  for (const budget of [-1, 0, 3]) assert.throws(() => s.scheduleChunked(budget));
  assert.equal(s.waiting.length, 1);
  assert.equal(s.running.length, 0);
  assert.equal(isEmpty(s.scheduleChunked(4)), false);
});

test("an idle step is empty and not pending", () => {
  const s = new Scheduler(config(200, 4));
  assert.ok(isEmpty(s.scheduleChunked(32)));
  assert.ok(isEmpty(s.scheduleChunked(32)));
  assert.throws(() => s.update());
});

test("a step is refused while the previous one is pending", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 8, 2);
  s.scheduleChunked(32);
  assert.throws(() => s.scheduleChunked(32));
  s.update();
  assert.equal(isEmpty(s.scheduleChunked(32)), false);
});

test("a long prompt spreads over steps and only the last chunk is sampled", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 100, 5);
  for (const first of [0, 32, 64]) {
    const step = run(s, 32);
    assert.deepEqual([step.decode, step.sampled, chunks(step)], [[], [], [[1, first, 32]]]);
    assert.equal(s.requests.get(1).output, 0);
  }
  const last = run(s, 32);
  assert.deepEqual([chunks(last), last.sampled], [[[1, 96, 4]], [1]]);
  const next = run(s, 32);
  assert.deepEqual([next.decode, next.prefill, numTokens(next)], [[1], [], 1]);
});

test("the whole length is allocated at admission", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 100, 5);
  const before = s.blocks.free;
  s.scheduleChunked(10);
  assert.equal(s.blocks.lengths.get(1), 100);
  assert.equal(s.blocks.free, before - 7);
  s.update();
  assert.equal(s.blocks.free, before - 7);
});

test("decode rows come first and the prefill takes the rest of the budget", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 4, 50);
  s.submit(2, 4, 50);
  run(s, 32);
  s.submit(3, 100, 5);
  const step = run(s, 32);
  assert.deepEqual([step.decode, chunks(step), step.sampled, numTokens(step)], [[1, 2], [[3, 0, 30]], [1, 2], 32]);
  assert.deepEqual(chunks(run(s, 32)), [[3, 30, 30]]);
});

test("one step can finish a prefill and start the next", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 10, 5);
  s.submit(2, 50, 5);
  const step = run(s, 32);
  assert.deepEqual([chunks(step), step.sampled, numTokens(step)], [[[1, 0, 10], [2, 0, 22]], [1], 32]);
});

test("the older prefill continues before a new request is admitted", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 50, 5);
  run(s, 32);
  s.submit(2, 50, 5);
  const second = run(s, 32);
  assert.deepEqual([chunks(second), second.decode, second.sampled], [[[1, 32, 18], [2, 0, 14]], [], [1]]);
});

test("admission stops at max_batch", () => {
  const s = new Scheduler(config(200, 2));
  for (const id of [1, 2, 3]) s.submit(id, 3, 50);
  assert.deepEqual(chunks(run(s, 32)), [[1, 0, 3], [2, 0, 3]]);
  assert.deepEqual([s.running.length, s.waiting.length], [2, 1]);
});

test("a budget of max_batch still advances the prefill", () => {
  const s = new Scheduler(config(200, 4));
  for (const id of [1, 2, 3]) s.submit(id, 1, 50);
  run(s, 4);
  s.submit(4, 3, 50);
  for (const first of [0, 1, 2]) {
    const step = run(s, 4);
    assert.deepEqual([step.decode, chunks(step)], [[1, 2, 3], [[4, first, 1]]]);
  }
  assert.equal(s.requests.get(4).output, 1);
});

test("the head of the queue is never skipped", () => {
  const s = new Scheduler(config(21, 4, 256));
  s.submit(1, 200, 50);
  s.submit(2, 200, 50);
  s.submit(3, 2, 50);
  assert.deepEqual(chunks(run(s, 32)), [[1, 0, 32]]);
  assert.equal(s.waiting.length, 2);
  assert.deepEqual(chunks(run(s, 512)), [[1, 32, 168]]);
  assert.equal(s.waiting.length, 2);
});

test("max_prefill_tokens does not limit a chunked step", () => {
  const s = new Scheduler(config(200, 4, 256));
  for (const id of [1, 2, 3]) s.submit(id, 200, 5);
  assert.deepEqual(chunks(run(s, 512)), [[1, 0, 200], [2, 0, 200], [3, 0, 112]]);
});

test("the partly prefilled sequence is preempted first and starts again at 0", () => {
  const s = new Scheduler(config(17, 4, 256));
  s.submit(1, 16, 100);
  run(s, 32);
  s.submit(2, 14 * 16, 5);
  const first = run(s, 4);
  assert.deepEqual([first.decode, chunks(first)], [[1], [[2, 0, 3]]]);
  assert.equal(s.blocks.free, 0);
  let guard = 0;
  let step;
  do {
    step = run(s, 4);
    assert.ok(++guard < 100);
  } while (s.requests.get(2).preemptions === 0);
  assert.deepEqual([s.requests.get(2).running, s.waiting, s.requests.get(2).output], [false, [2], 0]);
  assert.deepEqual([step.decode, step.prefill, s.requests.get(1).preemptions], [[1], [], 0]);
  assert.ok(!s.progress.has(2));
  s.running = [];
  s.blocks.release(1);
  s.requests.delete(1);
  assert.deepEqual(chunks(run(s, 4)), [[2, 0, 4]]);
});

test("a preempted decoder replays its prompt and its output", () => {
  const s = new Scheduler(config(5, 4, 64));
  s.submit(1, 8, 50);
  s.submit(2, 16, 40);
  run(s, 32);
  let guard = 0;
  while (s.requests.get(2).preemptions === 0) {
    run(s, 32);
    assert.ok(++guard < 100);
  }
  const length = s.requests.get(2).prompt + s.requests.get(2).output;
  assert.ok(s.requests.get(2).output > 0);
  s.running = s.running.filter((x) => x !== 1);
  s.blocks.release(1);
  const step = s.scheduleChunked(4);
  assert.deepEqual([chunks(step), step.sampled, s.blocks.lengths.get(2)], [[[2, 0, 4]], [], length]);
});

test("a prompt equal to the budget finishes in one step", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 32, 5);
  const first = run(s, 32);
  assert.deepEqual([chunks(first), first.sampled], [[[1, 0, 32]], [1]]);
  assert.deepEqual(run(s, 32).decode, [1]);
});

test("a step never exceeds the budget and the blocks always add up", () => {
  for (const budget of [8, 9, 32, 256]) {
    const s = new Scheduler(config(60, 8, 256));
    for (let i = 0; i < 12; i++) s.submit(i, 1 + ((i * 53) % 180), 1 + ((i * 29) % 60));
    for (let n = 0; n < 4000 && s.waiting.length + s.running.length > 0; n++) {
      const step = run(s, budget);
      assert.ok(numTokens(step) <= budget);
      assert.ok(s.running.length <= 8);
      const owned = [...s.blocks.tables.values()].reduce((a, b) => a + b, 0);
      assert.equal(s.blocks.free + owned, 59);
      if (step.prefill.some((c) => s.progress.has(c.id))) assert.equal(numTokens(step), budget);
    }
    assert.equal(s.waiting.length + s.running.length, 0, `budget ${budget}`);
    assert.equal(s.blocks.free, 59);
  }
});

test("schedule() admits whole prompts in one prefill-only step", () => {
  const s = new Scheduler(config(200, 4));
  s.submit(1, 100, 5);
  s.submit(2, 30, 5);
  const step = s.schedule();
  assert.deepEqual([step.prefill, step.seqs], [true, [1, 2]]);
  s.update();
  assert.deepEqual([s.requests.get(1).output, s.requests.get(2).output], [1, 1]);
  const next = s.schedule();
  assert.deepEqual([next.prefill, next.seqs], [false, [1, 2]]);
});

test("schedule() stops admitting at max_prefill_tokens and at max_batch", () => {
  const s = new Scheduler(config(400, 2, 256));
  s.submit(1, 200, 5);
  s.submit(2, 100, 5);
  assert.deepEqual(s.schedule().seqs, [1]);
  s.update();
  assert.deepEqual(s.schedule().seqs, [2]);
  s.update();
  s.submit(3, 4, 5);
  const decode = s.schedule();
  assert.deepEqual([decode.prefill, decode.seqs, s.waiting], [false, [1, 2], [3]]);
});

test("schedule() preempts from the back of the running list when a decoder needs a block", () => {
  const s = new Scheduler(config(4, 2, 32));
  s.submit(1, 16, 50);
  s.submit(2, 16, 50);
  s.submit(3, 16, 50);
  s.schedule();
  s.update();
  assert.deepEqual([s.running, s.blocks.free], [[1, 2], 1]);
  const step = s.schedule();
  assert.deepEqual([step.prefill, step.seqs, s.waiting, s.running], [false, [1], [2, 3], [1]]);
  assert.deepEqual([s.requests.get(3).preemptions, s.requests.get(2).preemptions, s.requests.get(1).preemptions], [0, 1, 0]);
});

test("playground: a 2,048-token prompt at budget 256 takes 9 steps to its first token and never stalls the decoders", () => {
  const { legacy, chunked } = predict("roomy", 2048, 256);
  assert.deepEqual([chunked.ttft.steps, chunked.longestStall, chunked.biggest, chunked.preemptions], [9, 0, 256, 0]);
  assert.deepEqual([legacy.ttft.steps, legacy.longestStall, legacy.biggest], [1, 1, 2048]);
});

test("playground: a budget that holds the whole prompt still leaves the decode rows first", () => {
  const { chunked } = predict("roomy", 2048, 2048);
  assert.deepEqual([chunked.ttft.steps, chunked.biggest, chunked.longestStall], [2, 2048, 0]);
});

test("playground: every decoder row appears in each chunked step and the lanes line up with the steps", () => {
  const g = new Playground("roomy");
  g.drop(512);
  for (let i = 0; i < 6; i++) g.advance(128);
  const run = g.runs["schedule_chunked()"];
  assert.deepEqual(run.steps.map((s) => s.decode), [4, 4, 4, 4, 4, 5]);
  assert.deepEqual(run.steps.map((s) => s.tokens), [128, 128, 128, 128, 20, 5]);
  const prompt = run.lanes.find((l) => l.kind === "prompt");
  assert.deepEqual(prompt.cells, ["chunk", "chunk", "chunk", "chunk", "first", "decode"]);
  assert.ok(run.lanes.every((l) => l.cells.length === 6));
  assert.equal(run.steps[0].tokens, 128);
  assert.equal(run.steps.reduce((n, s) => n + s.slices.reduce((m, c) => m + c.count, 0), 0), 512);
});

test("playground: under schedule() the decoders stall for the prefill step and the prompt gets its token at once", () => {
  const g = new Playground("roomy");
  g.drop(1024);
  g.advance(64);
  const run = g.runs["schedule()"];
  assert.deepEqual(run.lanes.map((l) => l.cells[0]), ["stall", "stall", "stall", "stall", "first"]);
  assert.deepEqual([run.stats().ttft.steps, run.stats().longestStall, run.steps[0].tokens], [1, 1, 1024]);
});

test("playground: a tight pool with a small budget preempts the prompt in prefill and restarts it", () => {
  const { chunked } = predict("tight", 2048, 32);
  assert.equal(chunked.preemptions, 1);
  assert.ok(chunked.ttft.steps > 74);
});

test("playground: prompts dropped later get their own lane that starts at the drop step", () => {
  const g = new Playground("roomy");
  g.advance(64);
  g.advance(64);
  g.drop(50);
  g.advance(64);
  const run = g.runs["schedule_chunked()"];
  const prompt = run.lanes[4];
  assert.deepEqual([prompt.from, prompt.cells, prompt.firstAt], [2, ["first"], 3]);
  assert.equal(run.stats().ttft.steps, 1);
});
