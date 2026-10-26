export function replayKv(trace, file) {
  const { kv_blocks: total, block_size: size } = trace.meta;
  const free = Array.from({ length: total }, (_, i) => i > 0);
  const prompt = new Map(trace.requests.map((r) => [r.id, r]));
  const live = new Map();
  const requests = [];
  const steps = [];
  let decodes = 0;
  let peak = { blocks: 0, step: 0, t: 0, running: 0, tokens: 0 };
  let maxBlock = 0;
  let maxRunning = 0;

  if (trace.requests.some((r) => r.preemptions > 0)) throw new Error(`${file}: the replay does not model preemption`);

  const take = (step) => {
    const id = free.indexOf(true, 1);
    if (id < 0) throw new Error(`${file}: step ${step} runs out of blocks`);
    free[id] = false;
    maxBlock = Math.max(maxBlock, id);
    return id;
  };

  for (const s of trace.steps) {
    if (s.kind === "prefill") {
      for (const id of s.seqs) {
        const r = prompt.get(id);
        if (!r || live.has(id)) throw new Error(`${file}: step ${s.index} admits sequence ${id} twice or unknown`);
        const seq = { id, prompt: r.prompt_tokens, output: r.output_tokens, arrival_s: r.arrival_s, admitted: s.index, decodes, finished: -1, length: r.prompt_tokens, blocks: [] };
        for (let b = 0; b < Math.ceil(seq.length / size); b++) seq.blocks.push({ id: take(s.index), step: s.index });
        live.set(id, seq);
        requests.push(seq);
      }
    } else {
      decodes++;
      for (const id of s.seqs) {
        const seq = live.get(id);
        if (!seq) throw new Error(`${file}: step ${s.index} decodes sequence ${id}, which holds no blocks`);
        if (seq.length % size === 0) seq.blocks.push({ id: take(s.index), step: s.index });
        seq.length++;
      }
    }
    maxRunning = Math.max(maxRunning, live.size);
    for (const id of s.finished) {
      const seq = live.get(id);
      if (!seq) throw new Error(`${file}: step ${s.index} finishes sequence ${id}, which holds no blocks`);
      for (const b of seq.blocks) free[b.id] = true;
      seq.finished = s.index;
      live.delete(id);
    }
    const held = total - 1 - free.filter(Boolean).length;
    if (total - 1 - held !== s.free_blocks) throw new Error(`${file}: step ${s.index} leaves ${total - 1 - held} blocks free, the trace says ${s.free_blocks}`);
    if (held > peak.blocks) {
      const tokens = [...live.values()].reduce((n, q) => n + q.length, 0);
      peak = { blocks: held, step: s.index, t: s.end_s, running: live.size, tokens };
    }
    steps.push([Math.round(s.start_s * 1e4) / 1e4, s.kind === "prefill" ? 1 : 0, held, decodes]);
  }

  return {
    file,
    meta: trace.meta,
    duration_s: trace.steps.length ? trace.steps[trace.steps.length - 1].end_s : 0,
    peak,
    max_block: maxBlock,
    max_running: maxRunning,
    requests: requests.map(({ length, ...r }) => ({ ...r, final_length: length })),
    steps,
  };
}
