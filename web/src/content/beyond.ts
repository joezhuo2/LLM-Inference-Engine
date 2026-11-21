import { beyondTitle, chapterNumber } from "./book-order";
import { estimates, fitEstimate } from "./estimates";
import { kernelStory } from "./kernels";
import { profilingTable } from "@/lib/data";
import { shardedForward } from "./kernels/tensor-parallelism";
import { more, spec, specFmt as fmt } from "./kernels/speculative-decoding";

export type BeyondSource = { label: string; href: string };
export type BeyondTopic = { id: string; title: string; unit: string; paragraphs: string[]; source: BeyondSource; estimate?: string };

const f = fitEstimate;
const estimateNumber = estimates.findIndex((e) => e.id === f.id) + 1;
const title = (slug: string) => kernelStory(slug)?.title ?? slug;

const forward = shardedForward();
const shardTable = profilingTable("tensor-parallelism-the-weights-of-one-rank", 0, "results");
const shardCell = (layout: string, name: string) => {
  const row = shardTable.rows.find((r) => r[0] === layout);
  const v = row ? Number(row[shardTable.header.indexOf(name)].replaceAll(",", "")) : Number.NaN;
  if (Number.isNaN(v)) throw new Error(`results.md Tensor parallelism: no "${name}" for ${layout}`);
  return v;
};
const shard = {
  pct: Math.round((shardCell("rank 0 of 2", "MiB on the device") / shardCell("full model", "MiB on the device")) * 100),
  loadMs: shardCell("rank 0 of 2", "upload ms"),
  fullMs: shardCell("full model", "upload ms"),
};
if (shard.pct <= 50 || shard.loadMs >= shard.fullMs) throw new Error("results.md Tensor parallelism: Beyond says one of two shares is more than half the model and loads faster than the whole model; rewrite the multiple GPUs topic");

export { beyondTitle };

export const beyondIntro = [
  `This book's argument fits in one sentence. At batch 1 a decode step reads every weight once, so tokens per second can be no higher than the memory bandwidth divided by the bytes read per token. Chapter ${chapterNumber("foundations")}, ${title("foundations")}, sets up that bound, and chapter ${chapterNumber("gemm")}, ${title("gemm")}, measures the projections against it.`,
  "The engine in this book stops at one model in BF16 on one GPU. Real engines go further in four ways, and each one changes the bytes read per token.",
  `Quantization and prefix caching are not implemented or measured in this engine (quantization is planned, and chapter ${chapterNumber("quantization")} has its arithmetic), and of multiple GPUs the engine has the split, checked on one GPU: chapter ${chapterNumber("tensor-parallelism")}, ${title("tensor-parallelism")}. Speculative decoding is, in its simplest form: chapter ${chapterNumber("speculative-decoding")}, ${title("speculative-decoding")}, guesses by copying from the context. The form with a draft model runs too, with a draft model that turned out too slow to pay. What follows says what each one does to the book's unit and what it costs, with one source to read for each.`,
];

export const beyondTopics: BeyondTopic[] = [
  {
    id: "quantization",
    title: "quantization",
    unit: "fewer bytes per weight",
    paragraphs: [
      `Quantization stores each weight in fewer bits, so a decode step reads fewer bytes and the batch-1 floor falls with them. It has its own chapter: chapter ${chapterNumber("quantization")}, ${title("quantization")}, works out what FP8 and 4-bit weights do to the floor, the arithmetic and the KV cache, and what the engine's FP8 stubs promise. None of it is measured yet. A real 4-bit file is a little larger than bits times weights, because each small block of weights also stores its scale.`,
      `Estimate ${estimateNumber} works one case on this book's GPU. ${f.model} is ${f.bf16} in BF16 and does not fit in ${f.vram} of VRAM. At ${f.lowBits} bits it is ${f.low}, and at the measured ${f.bandwidth} the floor is ${f.floorLow} per token, ${f.rateLow} tokens per second (calculated, not run).`,
      "GPTQ is one method for choosing the rounded values: it quantizes a trained model one layer at a time, using second-order information from a small set of calibration data, with no retraining.",
    ],
    source: { label: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)", href: "https://arxiv.org/abs/2210.17323" },
    estimate: f.id,
  },
  {
    id: "speculative-decoding",
    title: "speculative decoding with a draft model",
    unit: "the large model's bytes shared by several tokens",
    paragraphs: [
      `Chapter ${chapterNumber("speculative-decoding")} guesses the next tokens by copying them from earlier in the context, which works when the answer repeats its prompt and not otherwise. A draft model can guess anywhere.`,
      "A small draft model writes a few tokens ahead, one decode step each. The large model then scores all of them in one pass, which reads its weights once, the same way prefill handles many prompt tokens with one read. A round reads the draft's weights once per drafted token and the large model's weights once, and it yields the accepted tokens plus one more from the large model. So the large model's bytes are divided over the tokens of the round, where plain decoding pays all of them for every token.",
      "A drafted token is kept with a probability set by how the two models rated it, and a rejected one is replaced by a draw from a corrected distribution. With that rule the output has exactly the distribution of the large model alone. The gain depends on how often the draft is accepted: a draft that is usually wrong makes each kept token cost more bytes than plain decoding does, because the draft's reads are wasted.",
      `This engine can take its guesses from a draft model (measured, with TinyStories-110M drafting for TinyLlama): the large model agreed with ${fmt(more.draftAcceptedPct, 1)}% of 2 guesses per step, and the time per token rose from ${fmt(spec.chatPlain.ms, 2)} to ${fmt(more.draftMs, 2)} ms, ${fmt(more.draftSpeedup, 2)} times the tokens per second. That draft is a tenth of the large model's size but decodes only about 1.5 times as fast here (187 against about 125 tokens per second), so a guess costs most of a step. It also guesses greedily and is checked against the token the large model picks, not with the rule above.`,
    ],
    source: { label: "Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)", href: "https://arxiv.org/abs/2211.17192" },
  },
  {
    id: "prefix-caching",
    title: "prefix caching",
    unit: "no weight reads for a shared prefix",
    paragraphs: [
      "Many requests begin with the same tokens: a system prompt, a document, the earlier turns of a chat. The keys and values of a prefix depend only on the tokens in it, so they can be computed once and reused. A request that starts with a cached prefix skips prefill for those tokens. They cost no weight reads at all, and only the rest of the prompt is prefilled.",
      `The block tables of chapter ${chapterNumber("paged-kv")}, ${title("paged-kv")}, are the structure that makes this possible: a request finds its keys and values through a block table, and two tables can point at the same blocks. This engine does not do it. Its BlockManager marks each block free or used, a block belongs to one request, and all of a request's blocks are freed when it ends. Sharing needs a count of the requests that use each block, a copy when a request has to write into a shared block (copy on write), and a way to find the longest cached prefix of a new prompt, such as the radix tree of SGLang's RadixAttention.`,
    ],
    source: { label: "SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)", href: "https://arxiv.org/abs/2312.07104" },
  },
  {
    id: "multiple-gpus",
    title: "multiple GPUs",
    unit: "each GPU reads a share of the bytes",
    paragraphs: [
      "Tensor parallelism splits every layer's weight matrices across the GPUs. Each GPU reads only its share of the bytes for a token, and together the GPUs have several times the memory bandwidth. The price is communication: the partial results have to be combined inside every layer, on every step, over a link that is far slower than a GPU's own memory. It is also the way to run a model whose weights do not fit in one GPU.",
      "Pipeline parallelism puts different layers on different GPUs, and replicas put a whole copy of the model on each. In both, one token still has all of the weights read for it, one GPU at a time, so they raise the number of requests served and not the speed of one stream. The Megatron-LM paper describes the tensor split. It was written for training, and inference uses the same split of the forward pass.",
      `This engine cuts TinyLlama's weights for 2 or 4 GPUs and runs the forward pass on the shares, all of them taking turns on its one GPU (chapter ${chapterNumber("tensor-parallelism")}). A share of 2 is ${shard.pct}% of the model and not half, since every GPU keeps the embedding and the LM head whole, and it loads in ${shard.loadMs.toFixed(0)} ms against ${shard.fullMs.toFixed(0)} for the whole model. The shares give the whole model's tokens, and the logits come within ${forward.maxLogitDiff} (measured). On one GPU that is slower, ${forward.msPerToken[1].toFixed(2)} ms a token with 2 shares against ${forward.wholeMsPerToken.toFixed(2)}, so it checks the split and no more. The engine has no second GPU and no link between GPUs, and the cost of that link is the part that decides the gain.`,
    ],
    source: { label: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)", href: "https://arxiv.org/abs/1909.08053" },
  },
];

export const beyondTopic = (id: string) => beyondTopics.find((t) => t.id === id);
