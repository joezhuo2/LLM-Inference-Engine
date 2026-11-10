import { flashKernel, modelShape, profilingTable } from "@/lib/data";
import { type KernelStory, measured } from "./types";

const prefill = "prefill-against-prompt-length";
const kernels = "flashattention-prefill-kernels";

export const flashSystems = { paged: "engine, paged prefill", flash: "engine", graphs: "engine + graphs", chunked: "engine + graphs, budget 256", llama: "llama.cpp -ub 512", llamaWide: "llama.cpp -ub 2048" } as const;
export const flashPaths = { paged: "paged decode kernels", first: "flash, first kernel", tiled: "flash, tiled kernel" } as const;

export const flashLengths = profilingTable(prefill, 0, "results").header.slice(1);
export const flashLong = flashLengths[flashLengths.length - 1];
export const flashShort = flashLengths[0];

export const flashTtft = (system: string, prompt: string) => measured(prefill, { system }, prompt, 0, "results");
export const flashPerToken = (system: string, prompt: string) => measured(prefill, { system }, prompt, 1, "results");
export const flashPath = (path: string, prompt: string) => measured(kernels, { "prefill attention": path }, prompt, 0, "results");

export const flashFmt = (v: number) => v.toLocaleString("en-US");
export const flashTokens = (prompt: string) => Number(prompt.replace(/,/g, ""));

const kernel = flashKernel();
const shape = modelShape();

export const flashShape = {
  tile: kernel.tile,
  half: kernel.tile / 2,
  warps: kernel.warps_per_block,
  tests: kernel.tests,
  layouts: kernel.layouts,
  maxHeadDim: kernel.max_head_dim,
  headDim: shape.hidden / shape.heads,
  heads: shape.heads,
  kvHeads: shape.kv_heads,
  group: shape.heads / shape.kv_heads,
  lanes: 32,
  lanesPerRow: 32 / (kernel.tile / 2),
};

const paged = flashTtft(flashSystems.paged, flashLong);
const tiled = flashTtft(flashSystems.flash, flashLong);
const first = flashPath(flashPaths.first, flashLong);
const llama = flashTtft(flashSystems.llama, flashLong);
const perToken = flashLengths.map((p) => flashPerToken(flashSystems.flash, p));
const chunkedCost = (flashTtft(flashSystems.chunked, flashLong) / flashTtft(flashSystems.graphs, flashLong) - 1) * 100;

export const flashNumbers = {
  paged,
  tiled,
  first,
  llama,
  ratio: paged / tiled,
  firstLoss: (first / flashPath(flashPaths.paged, flashLong) - 1) * 100,
  behind: paged / llama,
  ahead: (1 - tiled / llama) * 100,
  perTokenLow: Math.min(...perToken),
  perTokenHigh: Math.max(...perToken),
  pagedPerTokenShort: flashPerToken(flashSystems.paged, flashShort),
  pagedPerTokenLong: flashPerToken(flashSystems.paged, flashLong),
  chunkedCost,
  chunkedMs: flashTtft(flashSystems.chunked, flashLong),
  graphsMs: flashTtft(flashSystems.graphs, flashLong),
};

export const flashPrefill: KernelStory = {
  slug: "flash-prefill",
  index: 12,
  title: "FlashAttention prefill",
  book: true,
  about: true,
  short: `Prefill ran the decode attention once per prompt token, each row walking its whole context alone. A tiled kernel gives one warp ${flashShape.tile} query rows, streams the keys past them ${flashShape.tile} at a time on the tensor cores, and keeps an online softmax per row.`,
  formula: "O = softmax(Q·Kᵀ / √d)·V, one 16 × 16 tile at a time",
  headline: {
    value: `${flashFmt(paged)} → ${flashFmt(tiled)} ms`,
    label: `time to first token of one ${flashLong}-token prompt, with the per-row decode attention and with the tiled kernel`,
  },
  sources: [
    { path: "src/kernels/flash_prefill.cu", label: "the tiled kernel" },
    { path: "include/engine/kernels/flash_prefill.h", label: "the interface" },
    { path: "src/runtime/paged_runner.cpp", label: "PagedRunner: chunk rows to this kernel" },
    { path: "src/sched/paged_batch.cpp", label: "chunk_starts and cached_lens" },
    { path: "scripts/prefill_ttft.py", label: "the measurement" },
  ],
  tests: [
    { path: "tests/gpu/test_flash_prefill.cpp", label: `${flashShape.tests} kernel tests` },
    { path: "tests/gpu/test_paged_runner.cpp", label: "the runner with either kernel" },
    { path: "tests/unit/test_paged_batch.cpp", label: "chunk fields of a batch" },
  ],
  testing: `The ${flashShape.tests} kernel tests were written before any kernel and passed unchanged against both versions. They compare with the naive attention on a contiguous copy of the context: chunk lengths on both sides of the tile size, prompts up to 8,192 tokens, cached prefixes, 16 chunks in one call, empty chunks, ${flashShape.layouts} head and block layouts, and Hugging Face's attention on every layer of a traced prompt. They also pin what a rewrite could break quietly: the first position is its own value bit for bit, scores large enough to overflow without the running maximum, NaN planted in every cache slot the kernel must not read, and outputs that are bit-identical alone or in a batch and whatever follows a row. In the runner, whole-prompt and chunked runs of four prompts give the same argmax with either kernel.`,
  excerpts: [
    {
      file: "src/kernels/flash_prefill.cu",
      from: "asm volatile(",
      to: '"f"(acc[2]), "f"(acc[3]));',
      caption: "The tensor core instruction, written as inline PTX: a 16 × 16 BF16 fragment times a 16 × 8 one, added to a 16 × 8 FP32 accumulator",
    },
    {
      file: "src/kernels/flash_prefill.cu",
      from: "for (int half = 0; half < 2; ++half) {",
      to: "mma_bf16(score[half], q[step], b0, b1);",
      caption: "Q·Kᵀ for one key tile: two halves of 8 keys, 16 dimensions per call, the query fragments already in registers",
    },
    {
      file: "src/kernels/flash_prefill.cu",
      from: "const float new_max = fmaxf(running_max[r], group_max(tile_max));",
      to: "running_max[r] = new_max;",
      caption: "The online softmax of one row: the maximum and the sum are reduced over the 4 lanes that hold the row, and the weights are rounded to BF16",
    },
    {
      file: "src/kernels/flash_prefill.cu",
      from: "const uint32_t p[4] = {",
      to: "pack_bf16(weight[0][2], weight[0][3]), pack_bf16(weight[1][2], weight[1][3])};",
      caption: "The weights a lane already holds are packed into the first operand of P·V: nothing moves between lanes",
    },
    {
      file: "src/kernels/flash_prefill.cu",
      from: "if (key > last_position) return;",
      to: "v_row = row + v_column;",
      caption: "Where a key comes from: nothing past the last row of the tile, the paged cache for the cached prefix, qkv for the chunk's own rows",
    },
  ],
  measurements: [
    {
      doc: "results",
      section: prefill,
      table: 0,
      caption: "TTFT in ms against prompt tokens, one request per run, median of 5 (min to max); engine is the tiled kernel, engine, paged prefill the per-row decode attention",
      command: "python scripts/prefill_ttft.py --llama-batched-bench <path> --token-budget 256",
    },
    {
      doc: "results",
      section: prefill,
      table: 1,
      caption: "the same medians divided by the prompt length, ms per prompt token",
      command: "python scripts/prefill_ttft.py --llama-batched-bench <path> --token-budget 256",
    },
    {
      doc: "results",
      section: kernels,
      table: 0,
      caption: "three attention paths for the same prefill, same session; the first kernel measured at commit 9cff334 (results/prefill/first_flash_kernel.csv)",
      command: "engine bench --prefill-attention paged for the per-row decode kernels; the default is the tiled kernel",
    },
  ],
  timeline: [
    {
      date: "Nov 9",
      prs: [184],
      branch: "kernel/flash-prefill-stub",
      version: "spec",
      author: "claude",
      title: `A spec and ${flashShape.tests} tests before any kernel`,
      text: `The interface (one call for every prefill chunk of a step, one block table row per chunk), the numerics, what a row may depend on, and ${flashShape.tests} tests, so the kernel had a definition of done before it existed.`,
    },
    {
      date: "Nov 9",
      prs: [185],
      branch: "kernel/flash-prefill-impl",
      version: "first kernel",
      author: "both",
      title: "The first kernel: one block per row and head",
      text: `Joe's kernel reads the chunk's own keys and values from qkv and keeps an online softmax, in the shape of the decode kernels: one thread block per row and query head, one key at a time, two barriers per key. All ${flashShape.tests} tests pass.`,
    },
    {
      date: "Nov 10",
      prs: [],
      branch: "runtime/flash-prefill-wiring",
      version: "wired",
      author: "claude",
      title: "Wired behind a switch, and slower",
      text: "PagedBatch gains chunk_starts and cached_lens, the runner copies one block table row per chunk and sends chunk rows to the new kernel when the switch is on. Measured that way, the first kernel was slower than the path it replaced, so the switch stayed off.",
      metric: {
        kind: "bars",
        label: `TTFT of one ${flashLong}-token prompt, ms`,
        unit: "ms",
        digits: 1,
        bars: [
          { label: "decode kernels", value: flashPath(flashPaths.paged, flashLong) },
          { label: "first kernel", value: first, highlight: true },
        ],
      },
    },
    {
      date: "Nov 10",
      prs: [],
      branch: "runtime/flash-prefill-wiring",
      version: "tiled",
      author: "claude",
      title: "The tiled kernel",
      text: `Written by Claude at Joe's request, as the kernel the spec asked for: one warp owns ${flashShape.tile} query rows of one head, keys stream past ${flashShape.tile} at a time through mma.sync on the tensor cores, an online softmax per row, FP32 accumulators in registers, no shared memory and no barriers. The ${flashShape.tests} tests pass unchanged.`,
    },
    {
      date: "Nov 10",
      prs: [],
      branch: "runtime/flash-prefill-wiring",
      version: "measured",
      author: "claude",
      title: "The default, and ahead of llama.cpp",
      text: `The tiled kernel becomes the default prefill attention and the prefill table is rerun with llama.cpp in the same session. Cost per prompt token stays between ${flashNumbers.perTokenLow} and ${flashNumbers.perTokenHigh} ms from ${flashShort} to ${flashLong} tokens. Not done: an Nsight Compute profile of the kernel, and a rerun of the load sweeps.`,
      metric: {
        kind: "bars",
        label: `TTFT of one ${flashLong}-token prompt, ms`,
        unit: "ms",
        digits: 1,
        bars: [
          { label: "decode kernels", value: paged },
          { label: "llama.cpp", value: llama },
          { label: "tiled kernel", value: tiled, highlight: true },
        ],
      },
    },
  ],
};
