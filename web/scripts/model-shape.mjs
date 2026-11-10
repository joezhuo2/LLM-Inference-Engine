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

export function parseGraphRule(benchText, headerText, batchText) {
  const number = (text, pattern, what) => {
    const match = pattern.exec(text);
    if (!match) throw new Error(`no ${what}`);
    return Number(match[1]);
  };
  return {
    steps: number(benchText, /int graph_steps = (\d+);/, "graph_steps default in src/app/bench_command.cpp"),
    split_blocks: number(headerText, /kSplitAttentionBlocks = (\d+);/, "kSplitAttentionBlocks in include/engine/sched/paged_batch.h"),
    max_splits: number(batchText, /kMaxSplits = (\d+);/, "kMaxSplits in src/sched/paged_batch.cpp"),
    min_split_tokens: number(batchText, /kMinSplitTokens = (\d+);/, "kMinSplitTokens in src/sched/paged_batch.cpp"),
  };
}
