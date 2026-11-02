export type SseEvent = { event: string; data: string };

export function createSseParser(onEvent: (e: SseEvent) => void) {
  let buffer = "";
  let event = "message";
  let data: string[] = [];

  const line = (text: string) => {
    if (text === "") {
      if (data.length) onEvent({ event, data: data.join("\n") });
      event = "message";
      data = [];
      return;
    }
    if (text.startsWith(":")) return;
    const colon = text.indexOf(":");
    const field = colon < 0 ? text : text.slice(0, colon);
    const value = colon < 0 ? "" : text.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  };

  return (chunk: string) => {
    buffer += chunk;
    let end = buffer.search(/\r\n|\r|\n/);
    while (end >= 0) {
      const size = buffer[end] === "\r" && buffer[end + 1] === "\n" ? 2 : 1;
      if (buffer[end] === "\r" && end + 1 === buffer.length) break;
      line(buffer.slice(0, end));
      buffer = buffer.slice(end + size);
      end = buffer.search(/\r\n|\r|\n/);
    }
  };
}

export async function readSse(body: ReadableStream<Uint8Array>, onEvent: (e: SseEvent) => void, signal?: AbortSignal) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const feed = createSseParser(onEvent);
  const cancel = () => reader.cancel().catch(() => {});
  signal?.addEventListener("abort", cancel);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      feed(decoder.decode(value, { stream: true }));
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}
