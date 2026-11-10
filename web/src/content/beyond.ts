import { beyondTitle, chapterNumber } from "./book-order";
import { estimates, fitEstimate } from "./estimates";
import { kernelStory } from "./kernels";

export type BeyondSource = { label: string; href: string };
export type BeyondTopic = { id: string; title: string; unit: string; paragraphs: string[]; source: BeyondSource; estimate?: string };

const f = fitEstimate;
const estimateNumber = estimates.findIndex((e) => e.id === f.id) + 1;
const title = (slug: string) => kernelStory(slug)?.title ?? slug;

export { beyondTitle };

export const beyondIntro = [
  `This book's argument fits in one sentence. At batch 1 a decode step reads every weight once, so tokens per second can be no higher than the memory bandwidth divided by the bytes read per token. Chapter ${chapterNumber("foundations")}, ${title("foundations")}, sets up that bound, and chapter ${chapterNumber("gemm")}, ${title("gemm")}, measures the projections against it.`,
  "The engine in this book stops at one model in BF16 on one GPU. Real engines go further in four ways, and each one changes the bytes read per token.",
  "None of the four is implemented or measured in this engine. What follows says what each one does to the book's unit and what it costs, with one source to read for each.",
];

export const beyondTopics: BeyondTopic[] = [
  {
    id: "quantization",
    title: "quantization",
    unit: "fewer bytes per weight",
    paragraphs: [
      `Quantization stores each weight in fewer bits. This engine keeps its weights in BF16, ${f.bf16BytesPerWeight} bytes each, so every decode step reads ${f.bf16BytesPerWeight} bytes per weight. At ${f.lowBits} bits a step reads ${f.fewerBytes} times fewer bytes, and the batch-1 floor falls by the same factor, because the floor is bytes divided by bandwidth (calculated). It is the biggest single way to cut the bytes.`,
      `Estimate ${estimateNumber} works one case on this book's GPU. ${f.model} is ${f.bf16} in BF16 and does not fit in ${f.vram} of VRAM. At ${f.lowBits} bits it is ${f.low}, and at the measured ${f.bandwidth} the floor is ${f.floorLow} per token, ${f.rateLow} tokens per second (calculated, not run).`,
      "There are three costs. A real file is a little larger than bits times weights, because each group of weights also stores a scale factor. The kernel has to turn the small integers back into numbers before it multiplies, which adds work to a step that had almost none besides reading. And rounding changes the model's output, so the accuracy has to be measured. GPTQ is one method for choosing the rounded values: it quantizes a trained model one layer at a time, using second-order information from a small set of calibration data, with no retraining.",
    ],
    source: { label: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)", href: "https://arxiv.org/abs/2210.17323" },
    estimate: f.id,
  },
  {
    id: "speculative-decoding",
    title: "speculative decoding",
    unit: "the large model's bytes shared by several tokens",
    paragraphs: [
      "A small draft model writes a few tokens ahead, one decode step each. The large model then scores all of them in one pass, which reads its weights once, the same way prefill handles many prompt tokens with one read. A round reads the draft's weights once per drafted token and the large model's weights once, and it yields the accepted tokens plus one more from the large model. So the large model's bytes are divided over the tokens of the round, where plain decoding pays all of them for every token.",
      "A drafted token is kept with a probability set by how the two models rated it, and a rejected one is replaced by a draw from a corrected distribution. With that rule the output has exactly the distribution of the large model alone. The gain depends on how often the draft is accepted: a draft that is usually wrong makes every round read more bytes than plain decoding does.",
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
    ],
    source: { label: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)", href: "https://arxiv.org/abs/1909.08053" },
  },
];

export const beyondTopic = (id: string) => beyondTopics.find((t) => t.id === id);
