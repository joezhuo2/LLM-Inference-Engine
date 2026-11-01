import type { Metric } from "@/content/build-log";
import { type Author, measured } from "@/content/kernels/types";

export type DeadEndKind = "slower" | "broke" | "cliff" | "misread";

export const kindLabel: Record<DeadEndKind, string> = {
  slower: "slower than what it replaced",
  broke: "broke a correctness rule",
  cliff: "helped, then stopped helping",
  misread: "measured right, read wrong",
};

export type DeadEnd = {
  slug: string;
  kernel: string;
  date: string;
  prs: number[];
  branch: string;
  author: Author;
  kind: DeadEndKind;
  title: string;
  tried: string;
  happened: string;
  kept: string;
  source: { label: string; path: string };
  metric?: Metric;
};

const profiling = (section: string, label: string) => ({ label, path: `docs/profiling.md#${section}` });

const rope = "fast-rope-branch-31";
const fused = "fused-add-rmsnorm-branch-30";
const tuned = "tuning-add-rmsnorm-branch-39";
const small = "small-batches-branch-33";
const swiglu = "fast-swiglu-branch-32";
const split = "split-context-attention-branch-40";
const wired = "split-attention-in-the-forward-pass-branch-42";
const graphs = "split-attention-in-cuda-graphs-branch-43";
const occupancy = "split-attention-occupancy-branch-66";
const sampling = "sampling-branch-65";

const longPrompt = { "max batch": "1", prompt: "1,984" };

export const deadEnds: DeadEnd[] = [
  {
    slug: "add-rmsnorm",
    kernel: "add + RMSNorm",
    date: "Oct 15",
    prs: [39],
    branch: "kernel/add-rmsnorm-impl",
    author: "joe",
    kind: "slower",
    title: "Fusing two kernels saved bytes, not time",
    tried: "One kernel for the residual add and the RMSNorm after it, so the sum never makes a round trip through DRAM.",
    happened: "It moved two thirds of the bytes and was barely faster at decode sizes. Both versions were latency bound: eight scalar 2-byte loads per thread, a second pass that read the sums back, two barriers.",
    kept: "The fusion stayed for the launches it removed. The time came in Branch 39, from 128-bit loads, the row kept in registers and a shuffle reduction.",
    source: profiling(fused, "Fused add_rmsnorm"),
    metric: {
      kind: "bars",
      label: "one token, µs per call",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "add + norm", value: measured(fused, { tokens: "1" }, "naive total (us)") },
        { label: "first fused", value: measured(fused, { tokens: "1" }, "fast (us)") },
        { label: "tuned", value: measured(tuned, { tokens: "1" }, "tuned fused (us)"), highlight: true },
      ],
    },
  },
  {
    slug: "rope",
    kernel: "RoPE",
    date: "Oct 16",
    prs: [40],
    branch: "kernel/rope-vec-impl",
    author: "joe",
    kind: "slower",
    title: "A faster RoPE that was slower",
    tried: "One thread per dimension pair, walking all 36 heads of its token one after another so each angle is computed once.",
    happened: "Each head was a dependent global round trip, so the kernel took a flat 21 µs at every size, more than twice naive at decode sizes. It only won at 512 tokens, where there was enough work to hide the chain.",
    kept: "The final kernel spreads the heads over 8 thread groups and still computes each sine and cosine once per pair.",
    source: profiling(rope, "Fast RoPE"),
    metric: {
      kind: "bars",
      label: "one token, µs per call",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "naive", value: measured(rope, { tokens: "1" }, "naive (us)") },
        { label: "first draft", value: measured(rope, { tokens: "1" }, "first draft (us)") },
        { label: "final", value: measured(rope, { tokens: "1" }, "fast (us)"), highlight: true },
      ],
    },
  },
  {
    slug: "rope",
    kernel: "RoPE",
    date: "Oct 16",
    prs: [40],
    branch: "kernel/rope-vec-impl",
    author: "claude",
    kind: "broke",
    title: "One bit off in a frequency",
    tried: "Computing the rotation frequencies 1 / θ^(2i / d) on the device with __powf, then with powf.",
    happened: "Both differ from the host's std::pow in the last bit for some frequencies, and one ulp in a cosine moves many outputs at large positions. 3 of the 9 bit-identity tests failed.",
    kept: "The launcher computes the frequency table on the host exactly as the naive kernel does and passes it by value.",
    source: { label: "changelog: kernel/rope-vec-impl", path: "changelog.md" },
    metric: { kind: "value", value: "3 / 9", label: "bit-identity tests failed with device powf" },
  },
  {
    slug: "swiglu",
    kernel: "SwiGLU",
    date: "Oct 16",
    prs: [42, 43],
    branch: "kernel/swiglu-small-batch",
    author: "claude",
    kind: "slower",
    title: "Eight values per load, no gain at one token",
    tried: "128-bit loads of eight BF16 values per thread, which made long launches up to 2x faster.",
    happened: "A decode row is 5,632 outputs: 704 vectors, 3 blocks on a 26-SM GPU, each thread running 8 dependent exp chains. At one token it was slower than the first draft, whose 22 blocks spread the same chains wider.",
    kept: "Below about 5 tokens the kernel now takes the scalar path; the vector path starts where it wins.",
    source: profiling(small, "Small batches"),
    metric: {
      kind: "bars",
      label: "one token, µs per call",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "first draft", value: measured(swiglu, { tokens: "1" }, "first draft (us)") },
        { label: "vector", value: measured(swiglu, { tokens: "1" }, "fast (us)") },
        { label: "scalar path", value: measured(small, { tokens: "1" }, "fast (us)"), highlight: true },
      ],
    },
  },
  {
    slug: "attention-v2",
    kernel: "attention v2",
    date: "Oct 20",
    prs: [47],
    branch: "kernel/paged-attention-v2-impl",
    author: "joe",
    kind: "slower",
    title: "Warp shuffles in place of shared memory",
    tried: "Replacing v2's shared-memory tree reductions (the chunk maximum and the sum) with warp shuffles, usually the faster tool.",
    happened: "The kernel got 15 to 20% slower. The timings were not kept, only the result.",
    kept: "The shared-memory reductions. The shuffles stayed where they won: adding the 8 lanes' partial dot products.",
    source: profiling("paged-attention-v2-branch-37", "Paged attention v2"),
    metric: { kind: "value", value: "15–20%", label: "slower with shuffles (not tabulated)" },
  },
  {
    slug: "split-attention",
    kernel: "split attention",
    date: "Oct 23",
    prs: [51],
    branch: "kernel/paged-attention-split-impl",
    author: "claude",
    kind: "cliff",
    title: "Splitting short contexts",
    tried: "Cutting every context into ranges so more blocks run at once.",
    happened: "At 128 tokens a split holds too few keys to pay for the second launch and the partial writes: every split count lost to v2. With 16 rows the grid is already 512 blocks and splitting tied or lost up to 2,048 tokens.",
    kept: "A rule: split only with 1 to 4 rows and 512 tokens or more, at most 8 splits of at least 256 tokens.",
    source: profiling(split, "Split-context attention"),
    metric: {
      kind: "bars",
      label: "one row at 128 tokens, µs per call",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "v2", value: measured(split, { rows: "1", context: "128" }, "v2"), highlight: true },
        { label: "2 splits", value: measured(split, { rows: "1", context: "128" }, "2") },
        { label: "8 splits", value: measured(split, { rows: "1", context: "128" }, "8") },
      ],
    },
  },
  {
    slug: "split-attention",
    kernel: "split attention",
    date: "Oct 24",
    prs: [52],
    branch: "runtime/wire-attention-split",
    author: "claude",
    kind: "misread",
    title: "A speedup the graphs never saw",
    tried: "Wiring the split kernel into the decode step and measuring eager and CUDA Graph runs.",
    happened: "Eager mode got 1.4x faster at a 1,984-token prompt; the graph runs did not move. Each graph had been captured on a dummy batch with no context, where the rule says 1 split, so every replay launched v2.",
    kept: "One graph per batch size and split count, 11 instead of 5, in the next branch.",
    source: profiling(wired, "Split attention in the forward pass"),
    metric: {
      kind: "bars",
      label: "ms per token, 1,984-token prompt, batch 1",
      unit: "ms",
      digits: 1,
      bars: [
        { label: "split, eager", value: measured(wired, longPrompt, "split eager") },
        { label: "old graphs", value: measured(graphs, longPrompt, "main graphs") },
        { label: "fixed graphs", value: measured(graphs, longPrompt, "this branch graphs"), highlight: true },
      ],
    },
  },
  {
    slug: "split-attention",
    kernel: "split attention",
    date: "Oct 31",
    prs: [106],
    branch: "web/kernel-split-attention",
    author: "claude",
    kind: "cliff",
    title: "Eight splits barely beat four",
    tried: "Doubling the split count from 4 to 8 at one row and 2,048 tokens.",
    happened: "6 blocks fit per SM, so 156 fill one wave. 4 splits is 128 blocks; 8 is 256, and the last 100 run in a second, mostly empty wave. Each block had half the work and the time barely moved.",
    kept: "The rule's target of about 256 blocks, now with the measurement that explains its ceiling.",
    source: profiling(occupancy, "Split attention occupancy"),
    metric: {
      kind: "bars",
      label: "partial kernel, Nsight Compute, µs",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "4 splits", value: measured(occupancy, { kernel: "split partial", splits: "4" }, "time (us)") },
        { label: "8 splits", value: measured(occupancy, { kernel: "split partial", splits: "8" }, "time (us)") },
      ],
    },
  },
  {
    slug: "cuda-graphs",
    kernel: "CUDA Graphs",
    date: "Oct 26",
    prs: [67],
    branch: "docs/graphs-gap-reading",
    author: "claude",
    kind: "misread",
    title: "Blaming the CPU for the gaps",
    tried: "Explaining why CUDA Graphs helped: the first reading said the host could not launch kernels fast enough.",
    happened: "The Nsight Systems trace said otherwise: the CPU ran more than a millisecond ahead of the GPU, and the idle time was a gap of about 3 µs before every kernel on the GPU side, which a replayed graph removes.",
    kept: "The corrected reading in the profiling notes, and the rule to read the timeline before writing the explanation.",
    source: profiling("decode-cuda-graphs-branch-34", "Decode CUDA Graphs"),
    metric: { kind: "value", value: "3.3 → 0.28 µs", label: "median gap before a kernel, eager → replayed" },
  },
  {
    slug: "sampling",
    kernel: "sampling",
    date: "Oct 31",
    prs: [105],
    branch: "tools/sample-bench",
    author: "claude",
    kind: "slower",
    title: "The simplest sampling mode was the slowest",
    tried: "Timing every sampling mode against a full sort of the vocabulary.",
    happened: "Plain temperature sampling took longer than top-k: the Gumbel noise uses double-precision log for every one of 32,000 tokens, and FP64 is slow on a laptop GPU. With top-k only the 50 kept tokens get noise.",
    kept: "Nothing yet: a float version is 5 to 7x faster in a throwaway copy, but it changes the tokens a fixed seed draws, so it waits for a decision.",
    source: profiling(sampling, "Sampling"),
    metric: {
      kind: "bars",
      label: "one row, µs per call",
      unit: "µs",
      digits: 1,
      bars: [
        { label: "top-k 50", value: measured(sampling, { rows: "1" }, "top-k 50"), highlight: true },
        { label: "temperature", value: measured(sampling, { rows: "1" }, "temperature") },
      ],
    },
  },
];
