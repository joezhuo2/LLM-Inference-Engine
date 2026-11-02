import assert from "node:assert/strict";
import { test } from "node:test";
import { createSseParser, readSse } from "../src/lib/dashboard/sse.ts";

test("events split across chunks and line endings are put back together", () => {
  const seen = [];
  const feed = createSseParser((e) => seen.push(e));
  feed("event: step\r\ndata: {\"index\"");
  feed(":1}\r\n\r\n: comment\n\ndata: a\ndata: b\n");
  feed("\ndata: [DONE]\n\n");
  assert.deepEqual(seen, [
    { event: "step", data: '{"index":1}' },
    { event: "message", data: "a\nb" },
    { event: "message", data: "[DONE]" },
  ]);
});

test("a stream read to its end delivers every event", async () => {
  const bytes = new TextEncoder().encode('data: {"choices":[{"delta":{"content":" the"}}]}\n\ndata: [DONE]\n\n');
  const body = new ReadableStream({
    start(c) {
      c.enqueue(bytes.slice(0, 7));
      c.enqueue(bytes.slice(7));
      c.close();
    },
  });
  const seen = [];
  await readSse(body, (e) => seen.push(e.data));
  assert.deepEqual(seen, ['{"choices":[{"delta":{"content":" the"}}]}', "[DONE]"]);
});
