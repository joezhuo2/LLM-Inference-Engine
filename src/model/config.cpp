#include "engine/model/config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("config: " + what);
}

template <typename T>
T field(const json& j, const char* key) {
    if (!j.contains(key)) fail(std::string("missing field '") + key + "'");
    try {
        return j.at(key).get<T>();
    } catch (const json::type_error&) {
        fail(std::string("field '") + key + "' has the wrong type");
    }
}

template <typename T>
void require_if_present(const json& j, const char* key, const T& expected) {
    if (j.contains(key) && !j.at(key).is_null() && j.at(key).get<T>() != expected)
        fail(std::string("unsupported value for '") + key + "'");
}

}  // namespace

ModelConfig parse_config(std::string_view text) {
    json j;
    try {
        j = json::parse(text);
    } catch (const json::parse_error& e) {
        fail(std::string("invalid JSON: ") + e.what());
    }
    if (!j.is_object()) fail("top level must be an object");

    if (field<std::string>(j, "model_type") != "llama")
        fail("only model_type 'llama' is supported");
    if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null())
        fail("rope_scaling is not supported");
    require_if_present<std::string>(j, "hidden_act", "silu");
    require_if_present(j, "attention_bias", false);
    require_if_present(j, "mlp_bias", false);
    require_if_present(j, "pretraining_tp", 1);

    ModelConfig c;
    c.hidden = field<int>(j, "hidden_size");
    c.layers = field<int>(j, "num_hidden_layers");
    c.heads = field<int>(j, "num_attention_heads");
    c.kv_heads = field<int>(j, "num_key_value_heads");
    c.intermediate = field<int>(j, "intermediate_size");
    c.vocab = field<int>(j, "vocab_size");
    c.max_positions = field<int>(j, "max_position_embeddings");
    c.rope_theta = field<double>(j, "rope_theta");
    c.rms_eps = field<double>(j, "rms_norm_eps");
    c.bos_id = field<int>(j, "bos_token_id");
    c.eos_id = field<int>(j, "eos_token_id");
    c.tie_embeddings = j.value("tie_word_embeddings", false);

    for (int v :
         {c.hidden, c.layers, c.heads, c.kv_heads, c.intermediate, c.vocab, c.max_positions})
        if (v <= 0) fail("sizes must be positive");
    if (c.rope_theta <= 0.0 || c.rms_eps <= 0.0)
        fail("rope_theta and rms_norm_eps must be positive");
    if (c.hidden % c.heads != 0) fail("hidden_size must be divisible by num_attention_heads");
    if (c.heads % c.kv_heads != 0)
        fail("num_attention_heads must be divisible by num_key_value_heads");
    c.head_dim = c.hidden / c.heads;
    require_if_present(j, "head_dim", c.head_dim);
    return c;
}

ModelConfig load_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail("cannot open " + path.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return parse_config(ss.str());
}

}  // namespace engine
