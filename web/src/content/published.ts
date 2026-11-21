export const billions = (params: number) => `${Number((params / 1e9).toPrecision(3))} billion`;

const h100 = { vendor: "NVIDIA", model: "H100 SXM", bandwidthTBs: 3.35, bf16TFlops: 989, memoryGB: 80 };
const llama2 = { vendor: "Meta", model: "Llama 2 7B", params: 6.74e9, layers: 32, kvHeads: 32, headDim: 128 };
const llama3 = { vendor: "Meta", model: "Llama 3 8B", params: 8.03e9, layers: 32, kvHeads: 8, headDim: 128, hidden: 4096, vocab: 128256, context: 8192 };

export const published = {
  h100: { ...h100, source: `published spec: ${h100.vendor} ${h100.model}, ${h100.bandwidthTBs} TB/s HBM3, ${h100.bf16TFlops} TFLOP/s dense BF16, ${h100.memoryGB} GB` },
  llama2: { ...llama2, source: `published spec: ${llama2.vendor} ${llama2.model}, ${billions(llama2.params)} parameters, ${llama2.layers} layers, ${llama2.kvHeads} KV heads, head dimension ${llama2.headDim}` },
  llama3: { ...llama3, source: `published spec: ${llama3.vendor} ${llama3.model}, ${billions(llama3.params)} parameters, ${llama3.layers} layers, ${llama3.kvHeads} KV heads, head dimension ${llama3.headDim}`, shapeSource: `published spec: ${llama3.vendor} ${llama3.model} config, hidden size ${llama3.hidden}, a ${llama3.vocab.toLocaleString("en-US")}-token vocabulary, a ${llama3.context.toLocaleString("en-US")}-token context` },
};
