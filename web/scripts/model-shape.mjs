const keys = {
  hidden: "hidden_size",
  intermediate: "intermediate_size",
  layers: "num_hidden_layers",
  heads: "num_attention_heads",
  kv_heads: "num_key_value_heads",
  vocab: "vocab_size",
  max_positions: "max_position_embeddings",
};

export function parseModelShape(cppText) {
  const block = /kTinyLlama = R"\(([\s\S]*?)\)";/.exec(cppText);
  if (!block) throw new Error("tests/unit/test_config.cpp has no kTinyLlama config");
  const config = JSON.parse(block[1]);
  const shape = {};
  for (const [key, field] of Object.entries(keys)) {
    if (!Number.isInteger(config[field])) throw new Error(`kTinyLlama has no integer ${field}`);
    shape[key] = config[field];
  }
  return shape;
}

export function parseRooflineContext(resultsText) {
  const match = /mean attended context ([\d.]+) tokens/.exec(resultsText);
  if (!match) throw new Error("docs/results.md has no mean attended context in its Roofline section");
  return Number(match[1]);
}
