import { kvReplay, modelShape } from "@/lib/data";
import { repoUrl } from "./build-log";
import { flashFmt, flashLong, flashProfile, flashRoofline, flashShape } from "./kernels/flash-prefill";
import { type Author, authorLabel } from "./kernels/types";

export type Runs = "cpu" | "gpu";

export type Numpy = { name: string; compares: string };

type Base = { id: string; chapter: string; title: string; task: string; nudge: string; runs: Runs };

export type StubExercise = Base & {
  kind: "stub";
  model: boolean;
  start: string;
  file: string;
  tests: string;
  filter: string;
  stubPrs: number[];
  implPrs: number[];
  end: string;
  author: Author;
  numpy?: Numpy;
};

export type OpenStart = { title: string; text: string; commands: string[] };

export type OpenExercise = Base & { kind: "open"; runs: "gpu"; check: string; compare: { path: string; label: string }[]; write?: OpenStart; tag?: string };

export const openTag = (e: OpenExercise) => e.tag ?? "no tests, open ended";

export type Exercise = StubExercise | OpenExercise;

export const exercises: Exercise[] = [
  {
    kind: "open",
    id: "bandwidth",
    chapter: "foundations",
    title: "measure the bandwidth roof",
    task: "Write a CUDA kernel that copies one large array into another, time it with CUDA events and turn the time into GB/s. That number is the ceiling every decode step on your GPU is measured against.",
    nudge: "Use arrays far larger than the L2 cache (hundreds of MB), run one warm-up launch before the timed ones, and count both the bytes read and the bytes written.",
    runs: "gpu",
    check: "There are no tests: check the copy is correct, then compare your GB/s with the memory bandwidth on your GPU's spec sheet. Joe's laptop reached 84% of its theoretical peak.",
    compare: [
      { path: "src/practice/bandwith_test.cu", label: "Joe's bandwidth test" },
      { path: "docs/exercises/bandwith_test.md", label: "his notes and result" },
    ],
  },
  {
    kind: "stub",
    id: "rope",
    chapter: "rope",
    title: "fast RoPE",
    task: "Rotate the queries and keys by their positions with a kernel that keeps every rotation pair busy at once, and matches Hugging Face bit for bit.",
    nudge: "Give each thread one rotation pair rather than one head. Read the spec for which two values form a pair and where the angles come from before writing any code.",
    runs: "gpu",
    model: false,
    start: "efecbaa",
    file: "src/kernels/rope_fast.cu",
    tests: "tests/todo/gpu/test_rope_fast.cpp",
    filter: "^FastRope\\.",
    stubPrs: [35],
    implPrs: [40],
    end: "15569cd",
    author: "both",
    numpy: { name: "rope", compares: "Your function against a loop that rotates each pair of values by the cosine and sine of position × theta^(−2i / head_dim), pairing value i with value i + head_dim / 2. The whole row is compared, so the V columns must come back unchanged." },
  },
  {
    kind: "stub",
    id: "block-manager",
    chapter: "paged-kv",
    title: "BlockManager",
    task: "Write the bookkeeping of the paged KV cache: hand out 16-token blocks to sequences as they grow, take them back when they finish, and turn a token's position into the slot that holds it.",
    nudge: "A list of free blocks plus, for each sequence, its length and its block table is enough. A sequence needs a new block only when its length is a multiple of 16 just before the append.",
    runs: "cpu",
    model: false,
    start: "135b3bd",
    file: "src/kv/block_manager.cpp",
    tests: "tests/todo/unit/test_block_manager.cpp",
    filter: "^BlockManager\\.",
    stubPrs: [24],
    implPrs: [29],
    end: "37a4712",
    author: "both",
  },
  {
    kind: "stub",
    id: "scheduler",
    chapter: "serving",
    title: "Scheduler",
    task: "Decide, step by step, which requests run together: admit waiting requests when their blocks fit, grow every running one by a token, free the finished ones and preempt the newest when the pool runs out.",
    nudge: "Keep a waiting queue and a running list and write admit, grow and finish as separate small steps. When a running sequence cannot grow, the newest running request gives back its blocks and goes to the front of the queue.",
    runs: "cpu",
    model: false,
    start: "f0d6978",
    file: "src/sched/scheduler.cpp",
    tests: "tests/todo/unit/test_scheduler.cpp",
    filter: "^Scheduler\\.",
    stubPrs: [28],
    implPrs: [33],
    end: "9a628be",
    author: "joe",
  },
  {
    kind: "stub",
    id: "write-kv",
    chapter: "paged-kv",
    title: "write_kv",
    task: "Write the kernel that copies each new token's keys and values out of the fused QKV row and into its slot in the paged cache.",
    nudge: "Give each thread one element: work out which token, head and dimension it is, then the slot gives the block (slot / 16) and the row inside it (slot % 16). A slot of -1 means write nothing.",
    runs: "gpu",
    model: false,
    start: "37a4712",
    file: "src/kernels/write_kv.cu",
    tests: "tests/todo/gpu/test_write_kv.cpp",
    filter: "^WriteKv\\.",
    stubPrs: [25],
    implPrs: [31],
    end: "f840222",
    author: "both",
    numpy: { name: "write_kv", compares: "Your caches against a copy filled by a loop, slot by slot. Every K and V value must land in block slot / block_size at row slot % block_size, rows with slot -1 write nothing, and every other cache value must stay as it was. The comparison is exact, because the kernel only copies." },
  },
  {
    kind: "stub",
    id: "paged-attention",
    chapter: "paged-attention",
    title: "paged attention, v1",
    task: "Write decode attention over the paged cache: one query per sequence and head against every key it has seen so far, found through the block table.",
    nudge: "One thread block per sequence and head. Write every score into shared memory, take the max, exponentiate and sum, then weight the value rows. Correct and simple first; fast comes in v2.",
    runs: "gpu",
    model: true,
    start: "f840222",
    file: "src/kernels/paged_attention.cu",
    tests: "tests/todo/gpu/test_paged_attention.cpp",
    filter: "^Paged(AttentionDecode|LayoutTest)\\.",
    stubPrs: [26],
    implPrs: [32],
    end: "f0d6978",
    author: "both",
    numpy: { name: "paged_attention", compares: "Your output against softmax(q · Kᵀ / √head_dim) · V, computed in float64 after gathering each row's blocks through its block table and cutting them to its context length. The cache holds NaN in every slot the context does not reach, so a read past the context shows. One row has no context and must stay zero. One has scores in the hundreds, where exp overflows unless you subtract the maximum first." },
  },
  {
    kind: "stub",
    id: "attention-v2",
    chapter: "attention-v2",
    title: "paged attention, v2",
    task: "Rewrite decode attention so it no longer keeps every score in shared memory, which lets it reach 8,192 tokens of context.",
    nudge: "Online softmax: keep a running max and a running sum, and when a new score raises the max, rescale what you have accumulated so far before adding to it.",
    runs: "gpu",
    model: true,
    start: "7fd2390",
    file: "src/kernels/paged_attention_v2.cu",
    tests: "tests/todo/gpu/test_paged_attention_v2.cpp",
    filter: "V2",
    stubPrs: [45],
    implPrs: [47],
    end: "2df22f6",
    author: "joe",
    numpy: { name: "paged_attention_v2", compares: "The same reference as v1. Your function gets a chunk size and must walk the context chunk by chunk with a running maximum and a running sum, rescaling what it has accumulated whenever the maximum rises. It is run with three chunk sizes, the smallest forcing many rescales. One row has scores that rise steadily along the context, which moves the maximum in every chunk and exposes a missing rescale." },
  },
  {
    kind: "stub",
    id: "split-attention",
    chapter: "split-attention",
    title: "split attention",
    task: "Cut each sequence's context into pieces that run on different SMs at once, then merge the partial results into one, so a few long sequences can fill the whole GPU.",
    nudge: "Each split runs v2's loop over its own slice and writes out its max, its sum and its partial output. A second kernel merges the splits with the same rescaling online softmax uses.",
    runs: "gpu",
    model: true,
    start: "e2a6214",
    file: "src/kernels/paged_attention_split.cu",
    tests: "tests/todo/gpu/test_paged_attention_split.cpp",
    filter: "Split",
    stubPrs: [50],
    implPrs: [51],
    end: "b597d93",
    author: "claude",
    numpy: { name: "paged_attention_split", compares: "The same reference as v1. Your function gets a split count and must cut each row's blocks into that many contiguous pieces (the file gives you split_ranges), keep a maximum, a sum and a partial output for each piece, then merge them with the same rescaling. It is run with several split counts, including more splits than a short row has blocks, so some splits are empty and must contribute nothing." },
  },
  {
    kind: "stub",
    id: "add-rmsnorm",
    chapter: "add-rmsnorm",
    title: "fused add + RMSNorm",
    task: "Add the residual and normalise the result in one kernel instead of two, so the row is read from memory once.",
    nudge: "One thread block per row. Add as you load, keep the sum of squares as you go, reduce it across the block, then scale each value by the weight and the inverse root mean square.",
    runs: "gpu",
    model: false,
    start: "8daf61d",
    file: "src/kernels/add_rmsnorm.cu",
    tests: "tests/todo/gpu/test_add_rmsnorm.cpp",
    filter: "^AddRmsnorm",
    stubPrs: [35],
    implPrs: [39],
    end: "efecbaa",
    author: "joe",
    numpy: { name: "add_rmsnorm", compares: "Both outputs against float64 formulas: sum = x + residual and norm = weight · sum / √(mean(sum²) + eps), with eps taken from the TinyLlama config. The rows include all zeros, which must not give NaN, and one so small that eps dominates the mean, so a wrong eps shows. It runs at the model's hidden size and at one that is not a multiple of 8." },
  },
  {
    kind: "stub",
    id: "swiglu",
    chapter: "swiglu",
    title: "fast SwiGLU",
    task: "Compute silu(gate) × up for every value of the MLP's middle layer at close to the memory bandwidth roof.",
    nudge: "Every output depends on two inputs and nothing else, so the work is all memory traffic. Let each thread handle several neighbouring values with wide loads.",
    runs: "gpu",
    model: false,
    start: "efecbaa",
    file: "src/kernels/swiglu_fast.cu",
    tests: "tests/todo/gpu/test_swiglu_fast.cpp",
    filter: "^(FastSwiglu|SwigluShapeTest)\\.",
    stubPrs: [35],
    implPrs: [42],
    end: "bb583b1",
    author: "both",
    numpy: { name: "swiglu", compares: "Your output against silu(gate) · up with silu(g) = g / (1 + e^−g), where the first half of each input row is the gate and the second half is up. Swapping the halves fails the check." },
  },
  {
    kind: "stub",
    id: "cuda-graphs",
    chapter: "cuda-graphs",
    title: "DecodeGraphs",
    task: "Capture a whole decode step as a CUDA graph and replay it, so the GPU no longer waits on the CPU to launch every kernel of the step one by one.",
    nudge: "Capture one graph per padded batch size. A graph replays the same pointers every time, so every input it reads has to live at a fixed device address that you copy into before each replay.",
    runs: "gpu",
    model: true,
    start: "b8fef56",
    file: "src/runtime/decode_graphs.cpp",
    tests: "tests/todo/gpu/test_decode_graphs.cpp",
    filter: "^DecodeGraphsTest\\.",
    stubPrs: [35],
    implPrs: [44],
    end: "7ce15d4",
    author: "joe",
  },
  {
    kind: "stub",
    id: "chunked-prefill",
    chapter: "chunked-prefill",
    title: "chunked prefill",
    task: "Teach the scheduler to cut a long prompt into pieces that share each step with the requests already decoding, so one long prompt no longer stalls everyone else.",
    nudge: "Fill each step's token budget with the decode rows first, one token each. Give what is left to the one prompt still in prefill, and only then admit new requests, first come, first served.",
    runs: "cpu",
    model: false,
    start: "2d9ee3b",
    file: "src/sched/scheduler.cpp",
    tests: "tests/todo/unit/test_scheduler_chunked.cpp",
    filter: "^SchedulerChunked\\.",
    stubPrs: [133],
    implPrs: [134],
    end: "cbffd5b",
    author: "both",
  },
  {
    kind: "open",
    id: "flash-shared-kv",
    chapter: "flash-prefill",
    title: `one read of K and V for ${flashShape.group} query heads`,
    task: `Change flash_attention_prefill so the ${flashShape.group} query heads that share a KV head load each key and value tile once, where today every warp loads it for itself. Predict from the roofline how far that moves the kernel, then measure it. Nobody has built this yet, so there is no solution to compare with and no measured speedup.`,
    nudge: `Count first: ${flashRoofline.now} FLOP per byte of K and V loaded today, ${flashRoofline.shared} if ${flashShape.group} heads share (calculated), against a ridge of ${flashRoofline.ridge.toFixed(0)}. The ${flashShape.warps} warps of a thread block already hold ${flashShape.warps} query heads of one KV head, so staging a tile once per block shares it ${flashShape.warps} ways without a larger block. All ${flashShape.group} need ${flashShape.group} warps per block, and at ${flashProfile.launchAfter.registers} registers per thread only ${flashRoofline.sharedBlocksPerSm} of those fit on an SM (calculated). Staging in shared memory also brings back a barrier per tile, which the kernel has none of today.`,
    runs: "gpu",
    tag: "open ended, the kernel's tests still apply",
    write: { title: "Change the kernel in place", text: "Start from main. The kernel is one file, and its tests are already part of the build.", commands: ["cmake --preset debug -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build --preset debug", "ctest --preset debug -R 'Flash(AttentionPrefill|Layout)'"] },
    check: `The ${flashShape.tests} tests of tests/gpu/test_flash_prefill.cpp must pass unchanged, the bit-for-bit ones included. Then profile one launch with the Nsight Compute command in docs/profiling.md and compare with today's kernel (measured): ${flashFmt(flashProfile.step.requestedMb)} MB requested, ${flashProfile.step.l1Hit}% of it in L1, tensor pipe at ${flashProfile.step.tensor}%, ${flashProfile.step.occupancy}% of warp places filled, ${flashProfile.step.ms} ms. The bound: at the measured GEMM peak the products of a whole ${flashLong}-token prefill would take ${flashRoofline.floorMs.toFixed(0)} ms of the ${flashRoofline.kernelMs} ms the kernel takes in Nsight Systems, so against that measured ceiling nothing gains more than ${flashRoofline.maxGain.toFixed(1)} times (calculated).`,
    compare: [
      { path: "src/kernels/flash_prefill.cu", label: "the kernel as it is today" },
      { path: "tests/gpu/test_flash_prefill.cpp", label: "the tests it must still pass" },
      { path: "docs/profiling.md", label: "the Nsight Compute numbers to start from, under FlashAttention prefill in chunks" },
    ],
  },
  {
    kind: "open",
    id: "speculative-break-even",
    chapter: "speculative-decoding",
    title: "find where guessing stops paying",
    task: "Write three prompts of your own: one whose answer copies most of the prompt, one that copies a little, and one that copies nothing. Run each with 0, 2, 4, 8 and 15 guesses per step and find the draft length where the time per token stops falling.",
    nudge: "Run engine bench with --max-batch 16 --graphs --num-requests 1 --stop-at-eos --prompts your.json --speculate K, and read the steps, the draft tokens and the accepted ones it prints. Then add --speculate-adaptive and compare the dropped rows.",
    runs: "gpu",
    check: "There are no tests: for each prompt, the tokens per step times the time of a step should give the time per token you measured. The prompt that copies nothing should cost about the same as plain decoding, never clearly less.",
    compare: [
      { path: "scripts/speculative_sweep.sh", label: "the sweep this chapter's table comes from" },
      { path: "scripts/prompts_repeat.json", label: "the four prompts whose answers repeat them" },
    ],
  },
];

export const chapterExercises = (slug: string): Exercise[] => exercises.filter((e) => e.chapter === slug);

export const runsLabel: Record<Runs, string> = { cpu: "runs on a CPU", gpu: "needs an NVIDIA GPU" };

export const numpyOf = (e: Exercise): Numpy | undefined => (e.kind === "stub" ? e.numpy : undefined);

export const cpuCount = () => exercises.filter((e) => e.runs === "cpu").length;

export const numpyCount = () => exercises.filter((e) => numpyOf(e)).length;

const numpyDir = "docs/exercises/numpy";

export const numpyFile = (n: Numpy) => `${numpyDir}/${n.name}.py`;

export const numpySolution = (n: Numpy) => `${numpyDir}/solutions/${n.name}.py`;

export const numpyCommand = (n: Numpy) => `python3 ${numpyFile(n)}`;

export const numpyUrl = (n: Numpy) => `${repoUrl}/blob/main/${numpyFile(n)}`;

export const numpyInstall = "pip install numpy";

export const numpyNote = "The engine keeps BF16 values and rounds at fixed points inside each kernel. The NumPy files use float32 and compare within a small tolerance, so they teach the algorithm and not the rounding; the CUDA tests make the exact comparison.";

export function numpyShapes(): string {
  const m = modelShape();
  const headDim = m.hidden / m.heads;
  return `Shapes are TinyLlama's: ${m.heads} query heads, ${m.kv_heads} key/value heads, head_dim ${headDim} (calculated: hidden size ${m.hidden.toLocaleString("en-US")} divided by ${m.heads} heads), MLP width ${m.intermediate.toLocaleString("en-US")} and ${kvReplay().meta.block_size}-token blocks. The files read the model sizes from the config in tests/unit/test_config.cpp; the block size is the engine's own.`;
}

export const cloneCommand = `git clone ${repoUrl}.git && cd LLM-Inference-Engine`;

export const prUrl = (n: number) => `${repoUrl}/pull/${n}`;

export function exerciseCommands(e: StubExercise): string[] {
  const gpu = e.runs === "gpu";
  const preset = gpu ? "debug" : "cpu";
  const configure = gpu ? `cmake --preset debug -DCMAKE_CUDA_ARCHITECTURES=native` : `cmake --preset cpu`;
  return [`git switch --detach ${e.start}`, `${configure} && cmake --build --preset ${preset}`, `ctest --preset ${preset} -L todo_joe -R '${e.filter}'`];
}

export const compareCommand = (e: StubExercise) => `git diff ${e.start} ${e.end} -- ${e.file}`;

export const implLabel = (e: StubExercise) => (e.author === "claude" ? "the implementation, drafted with Claude at Joe's request" : e.author === "joe" ? "Joe's implementation" : `the implementation (${authorLabel.both})`);
