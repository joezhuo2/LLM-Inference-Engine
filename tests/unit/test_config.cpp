#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "engine/model/config.h"
#include "support/paths.h"

using engine::ModelConfig;
using engine::parse_config;

namespace {

const std::string kTinyLlama = R"({
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "vocab_size": 32000
})";

std::string with(const std::string& key, const std::string& value) {
    const std::string needle = "\"" + key + "\": ";
    std::string s = kTinyLlama;
    const size_t start = s.find(needle);
    if (start == std::string::npos) return s.insert(1, needle + value + ",");
    const size_t begin = start + needle.size();
    const size_t end = s.find_first_of(",\n", begin);
    return s.replace(begin, end - begin, value);
}

std::string without(const std::string& key) {
    std::string s = kTinyLlama;
    const size_t start = s.find("\"" + key + "\"");
    const size_t end = s.find('\n', start);
    return s.erase(start, end - start + 1);
}

void expect_rejected(const std::string& json, const std::string& fragment) {
    try {
        parse_config(json);
        ADD_FAILURE() << "expected a rejection mentioning '" << fragment << "'";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

}  // namespace

TEST(Config, ParsesTinyLlama) {
    const ModelConfig c = parse_config(kTinyLlama);
    EXPECT_EQ(c.hidden, 2048);
    EXPECT_EQ(c.layers, 22);
    EXPECT_EQ(c.heads, 32);
    EXPECT_EQ(c.kv_heads, 4);
    EXPECT_EQ(c.head_dim, 64);
    EXPECT_EQ(c.intermediate, 5632);
    EXPECT_EQ(c.vocab, 32000);
    EXPECT_EQ(c.max_positions, 2048);
    EXPECT_DOUBLE_EQ(c.rope_theta, 10000.0);
    EXPECT_DOUBLE_EQ(c.rms_eps, 1e-5);
    EXPECT_EQ(c.bos_id, 1);
    EXPECT_EQ(c.eos_id, 2);
    EXPECT_FALSE(c.tie_embeddings);
}

TEST(Config, DerivedWidths) {
    const ModelConfig c = parse_config(kTinyLlama);
    EXPECT_EQ(c.kv_dim(), 256);
    EXPECT_EQ(c.qkv_dim(), 2560);
    EXPECT_EQ(c.group_size(), 8);
}

TEST(Config, OptionalFieldsMayBeAbsent) {
    for (const char* key :
         {"attention_bias", "hidden_act", "pretraining_tp", "rope_scaling", "tie_word_embeddings"})
        EXPECT_NO_THROW(parse_config(without(key))) << key;
    EXPECT_FALSE(parse_config(without("tie_word_embeddings")).tie_embeddings);
}

TEST(Config, AcceptsMatchingHeadDim) {
    EXPECT_EQ(parse_config(with("head_dim", "64")).head_dim, 64);
}

TEST(Config, RejectsMissingField) {
    expect_rejected(without("num_key_value_heads"), "num_key_value_heads");
    expect_rejected(without("rms_norm_eps"), "rms_norm_eps");
}

TEST(Config, RejectsWrongType) {
    expect_rejected(with("hidden_size", "\"2048\""), "hidden_size");
}

TEST(Config, RejectsUnsupportedArchitecture) {
    expect_rejected(with("model_type", "\"mistral\""), "model_type");
    expect_rejected(with("rope_scaling", R"({"type": "linear", "factor": 2.0})"), "rope_scaling");
    expect_rejected(with("hidden_act", "\"gelu\""), "hidden_act");
    expect_rejected(with("attention_bias", "true"), "attention_bias");
    expect_rejected(with("pretraining_tp", "2"), "pretraining_tp");
}

TEST(Config, RejectsInconsistentShapes) {
    expect_rejected(with("num_key_value_heads", "5"), "num_key_value_heads");
    expect_rejected(with("hidden_size", "2050"), "hidden_size");
    expect_rejected(with("head_dim", "128"), "head_dim");
    expect_rejected(with("num_hidden_layers", "0"), "positive");
}

TEST(Config, RejectsInvalidJson) {
    expect_rejected("{\"hidden_size\": ", "invalid JSON");
    expect_rejected("[1, 2]", "object");
}

TEST(Config, MissingFileThrows) {
    EXPECT_THROW(engine::load_config("/nonexistent/config.json"), std::runtime_error);
}

TEST(Config, LoadsTheRealTinyLlamaConfig) {
    const auto path = engine::test::model_dir() / "config.json";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
    const ModelConfig c = engine::load_config(path);
    EXPECT_EQ(c.hidden, 2048);
    EXPECT_EQ(c.layers, 22);
    EXPECT_EQ(c.kv_heads, 4);
    EXPECT_EQ(c.qkv_dim(), 2560);
    EXPECT_EQ(c.vocab, 32000);
    EXPECT_DOUBLE_EQ(c.rms_eps, 1e-5);
}
