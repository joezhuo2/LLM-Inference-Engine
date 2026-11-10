import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseGraphRule, parseModelShape, parseRooflineContext } from "./model-shape.mjs";

const cpp = `const std::string kTinyLlama = R"({
  "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22, "num_attention_heads": 32,
  "num_key_value_heads": 4, "vocab_size": 32000, "max_position_embeddings": 2048, "rope_scaling": null
})";`;

test("model shape is read from the C++ test fixture", () => {
  assert.deepEqual(parseModelShape(cpp), { hidden: 2048, intermediate: 5632, layers: 22, heads: 32, kv_heads: 4, vocab: 32000, max_positions: 2048 });
});

test("a missing field or fixture fails loudly", () => {
  assert.throws(() => parseModelShape(cpp.replace('"vocab_size": 32000,', "")), /vocab_size/);
  assert.throws(() => parseModelShape("nothing"), /kTinyLlama/);
});

test("the roofline context is read from results.md", () => {
  assert.equal(parseRooflineContext("Ceilings: 1 GB/s and 2 TFLOP/s (BF16), ridge point 3 FLOP/byte; mean attended context 96 tokens."), 96);
  assert.throws(() => parseRooflineContext("nothing"), /context/);
});

test("the repository files parse", () => {
  const shape = parseModelShape(readFileSync(new URL("../../tests/unit/test_config.cpp", import.meta.url), "utf8"));
  assert.equal(shape.hidden % shape.heads, 0);
  assert.ok(parseRooflineContext(readFileSync(new URL("../../docs/results.md", import.meta.url), "utf8")) > 0);
});

test("the graph rule is read from the C++ sources", () => {
  const rule = parseGraphRule("    int graph_steps = 2;", "inline constexpr int64_t kSplitAttentionBlocks = 256;", "constexpr int64_t kMaxSplits = 8;\nconstexpr int64_t kMinSplitTokens = 256;");
  assert.deepEqual(rule, { steps: 2, split_blocks: 256, max_splits: 8, min_split_tokens: 256 });
  assert.throws(() => parseGraphRule("nothing", "", ""), /graph_steps/);
  const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
  const repo = parseGraphRule(read("src/app/bench_command.cpp"), read("include/engine/sched/paged_batch.h"), read("src/sched/paged_batch.cpp"));
  assert.ok(repo.steps >= 1 && (repo.steps & (repo.steps - 1)) === 0);
});
