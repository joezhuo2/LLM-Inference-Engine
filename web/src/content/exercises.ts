import { repoUrl } from "./build-log";
import { type Author, authorLabel } from "./kernels/types";

type Base = { id: string; chapter: string; title: string; task: string; nudge: string };

export type StubExercise = Base & {
  kind: "stub";
  gpu: boolean;
  model: boolean;
  start: string;
  file: string;
  tests: string;
  filter: string;
  stubPrs: number[];
  implPrs: number[];
  end: string;
  author: Author;
};

export type OpenExercise = Base & { kind: "open"; gpu: true; check: string; compare: { path: string; label: string }[] };

export type Exercise = StubExercise | OpenExercise;

export const exercises: Exercise[] = [
  {
    kind: "open",
    id: "bandwidth",
    chapter: "foundations",
    title: "measure the bandwidth roof",
    task: "Write a CUDA kernel that copies one large array into another, time it with CUDA events and turn the time into GB/s. That number is the ceiling every decode step on your GPU is measured against.",
    nudge: "Use arrays far larger than the L2 cache (hundreds of MB), run one warm-up launch before the timed ones, and count both the bytes read and the bytes written.",
    gpu: true,
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
    gpu: true,
    model: false,
    start: "efecbaa",
    file: "src/kernels/rope_fast.cu",
    tests: "tests/todo/gpu/test_rope_fast.cpp",
    filter: "^FastRope\\.",
    stubPrs: [35],
    implPrs: [40],
    end: "15569cd",
    author: "both",
  },
  {
    kind: "stub",
    id: "block-manager",
    chapter: "paged-kv",
    title: "BlockManager",
    task: "Write the bookkeeping of the paged KV cache: hand out 16-token blocks to sequences as they grow, take them back when they finish, and turn a token's position into the slot that holds it.",
    nudge: "A list of free blocks plus, for each sequence, its length and its block table is enough. A sequence needs a new block only when its length is a multiple of 16 just before the append.",
    gpu: false,
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
    chapter: "paged-kv",
    title: "Scheduler",
    task: "Decide, step by step, which requests run together: admit waiting requests when their blocks fit, grow every running one by a token, free the finished ones and preempt the newest when the pool runs out.",
    nudge: "Keep a waiting queue and a running list and write admit, grow and finish as separate small steps. When a running sequence cannot grow, the newest running request gives back its blocks and goes to the front of the queue.",
    gpu: false,
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
    gpu: true,
    model: false,
    start: "37a4712",
    file: "src/kernels/write_kv.cu",
    tests: "tests/todo/gpu/test_write_kv.cpp",
    filter: "^WriteKv\\.",
    stubPrs: [25],
    implPrs: [31],
    end: "f840222",
    author: "both",
  },
  {
    kind: "stub",
    id: "paged-attention",
    chapter: "paged-attention",
    title: "paged attention, v1",
    task: "Write decode attention over the paged cache: one query per sequence and head against every key it has seen so far, found through the block table.",
    nudge: "One thread block per sequence and head. Write every score into shared memory, take the max, exponentiate and sum, then weight the value rows. Correct and simple first; fast comes in v2.",
    gpu: true,
    model: true,
    start: "f840222",
    file: "src/kernels/paged_attention.cu",
    tests: "tests/todo/gpu/test_paged_attention.cpp",
    filter: "^Paged(AttentionDecode|LayoutTest)\\.",
    stubPrs: [26],
    implPrs: [32],
    end: "f0d6978",
    author: "both",
  },
  {
    kind: "stub",
    id: "attention-v2",
    chapter: "attention-v2",
    title: "paged attention, v2",
    task: "Rewrite decode attention so it no longer keeps every score in shared memory, which lets it reach 8,192 tokens of context.",
    nudge: "Online softmax: keep a running max and a running sum, and when a new score raises the max, rescale what you have accumulated so far before adding to it.",
    gpu: true,
    model: true,
    start: "7fd2390",
    file: "src/kernels/paged_attention_v2.cu",
    tests: "tests/todo/gpu/test_paged_attention_v2.cpp",
    filter: "V2",
    stubPrs: [45],
    implPrs: [47],
    end: "2df22f6",
    author: "joe",
  },
  {
    kind: "stub",
    id: "split-attention",
    chapter: "split-attention",
    title: "split attention",
    task: "Cut each sequence's context into pieces that run on different SMs at once, then merge the partial results into one, so a few long sequences can fill the whole GPU.",
    nudge: "Each split runs v2's loop over its own slice and writes out its max, its sum and its partial output. A second kernel merges the splits with the same rescaling online softmax uses.",
    gpu: true,
    model: true,
    start: "e2a6214",
    file: "src/kernels/paged_attention_split.cu",
    tests: "tests/todo/gpu/test_paged_attention_split.cpp",
    filter: "Split",
    stubPrs: [50],
    implPrs: [51],
    end: "b597d93",
    author: "claude",
  },
  {
    kind: "stub",
    id: "add-rmsnorm",
    chapter: "add-rmsnorm",
    title: "fused add + RMSNorm",
    task: "Add the residual and normalise the result in one kernel instead of two, so the row is read from memory once.",
    nudge: "One thread block per row. Add as you load, keep the sum of squares as you go, reduce it across the block, then scale each value by the weight and the inverse root mean square.",
    gpu: true,
    model: false,
    start: "8daf61d",
    file: "src/kernels/add_rmsnorm.cu",
    tests: "tests/todo/gpu/test_add_rmsnorm.cpp",
    filter: "^AddRmsnorm",
    stubPrs: [35],
    implPrs: [39],
    end: "efecbaa",
    author: "joe",
  },
  {
    kind: "stub",
    id: "swiglu",
    chapter: "swiglu",
    title: "fast SwiGLU",
    task: "Compute silu(gate) × up for every value of the MLP's middle layer at close to the memory bandwidth roof.",
    nudge: "Every output depends on two inputs and nothing else, so the work is all memory traffic. Let each thread handle several neighbouring values with wide loads.",
    gpu: true,
    model: false,
    start: "efecbaa",
    file: "src/kernels/swiglu_fast.cu",
    tests: "tests/todo/gpu/test_swiglu_fast.cpp",
    filter: "^(FastSwiglu|SwigluShapeTest)\\.",
    stubPrs: [35],
    implPrs: [42],
    end: "bb583b1",
    author: "both",
  },
  {
    kind: "stub",
    id: "cuda-graphs",
    chapter: "cuda-graphs",
    title: "DecodeGraphs",
    task: "Capture a whole decode step as a CUDA graph and replay it, so the GPU no longer waits on the CPU to launch every kernel of the step one by one.",
    nudge: "Capture one graph per padded batch size. A graph replays the same pointers every time, so every input it reads has to live at a fixed device address that you copy into before each replay.",
    gpu: true,
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
    gpu: false,
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
];

export const chapterExercises = (slug: string): Exercise[] => exercises.filter((e) => e.chapter === slug);

export const cloneCommand = `git clone ${repoUrl}.git && cd LLM-Inference-Engine`;

export const prUrl = (n: number) => `${repoUrl}/pull/${n}`;

export function exerciseCommands(e: StubExercise): string[] {
  const preset = e.gpu ? "debug" : "cpu";
  const configure = e.gpu ? `cmake --preset debug -DCMAKE_CUDA_ARCHITECTURES=native` : `cmake --preset cpu`;
  return [`git switch --detach ${e.start}`, `${configure} && cmake --build --preset ${preset}`, `ctest --preset ${preset} -L todo_joe -R '${e.filter}'`];
}

export const compareCommand = (e: StubExercise) => `git diff ${e.start} ${e.end} -- ${e.file}`;

export const implLabel = (e: StubExercise) => (e.author === "claude" ? "the implementation, drafted with Claude at Joe's request" : e.author === "joe" ? "Joe's implementation" : `the implementation (${authorLabel.both})`);
