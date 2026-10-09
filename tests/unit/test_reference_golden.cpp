#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"
#include "engine/model/config.h"
#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/paths.h"
#include "support/reference.h"

using engine::DType;
using engine::test::ReferencePrompt;

namespace {

struct Expected {
    DType dtype;
    std::vector<int64_t> shape;
};

std::map<std::string, Expected> expected_tensors(const engine::ModelConfig& c,
                                                 const ReferencePrompt& p) {
    const int64_t t = p.total_len;
    const int64_t h = c.hidden;
    const int64_t q = int64_t(c.heads) * c.head_dim;
    std::map<std::string, Expected> e{
        {"input_ids", {DType::I32, {t}}},
        {"embed", {DType::BF16, {t, h}}},
        {"final_norm", {DType::BF16, {t, h}}},
        {"logits", {DType::BF16, {t, c.vocab}}},
    };
    for (int i = 0; i < c.layers; ++i) {
        const std::string l = "layers." + std::to_string(i) + ".";
        e[l + "out"] = {DType::BF16, {t, h}};
        if (!p.trace) continue;
        for (const char* name : {"ln1", "o", "ln2", "down"}) e[l + name] = {DType::BF16, {t, h}};
        for (const char* name : {"q", "attn"}) e[l + name] = {DType::BF16, {t, q}};
        for (const char* name : {"k", "v"}) e[l + name] = {DType::BF16, {t, c.kv_dim()}};
        for (const char* name : {"gate", "up", "act"})
            e[l + name] = {DType::BF16, {t, c.intermediate}};
    }
    return e;
}

}  // namespace

TEST(ReferenceGolden, FilesMatchTheManifestAndTokenizer) {
    const auto model = engine::test::model_dir();
    const auto manifest = engine::test::reference_dir() / "manifest.json";
    if (!std::filesystem::exists(model / "tokenizer.model")) GTEST_SKIP() << model << " not found";
    if (!std::filesystem::exists(manifest))
        GTEST_SKIP() << manifest << " not found, run scripts/dump_reference.py";

    const auto config = engine::load_config(model / "config.json");
    const engine::Tokenizer tok(model / "tokenizer.model");
    const auto prompts = engine::test::load_reference_manifest();
    ASSERT_FALSE(prompts.empty());
    int traced = 0;
    for (const ReferencePrompt& p : prompts) {
        SCOPED_TRACE(p.name);
        traced += p.trace;
        const engine::MappedFile file(p.file);
        const auto header = engine::parse_safetensors_header(file.bytes());

        const auto expected = expected_tensors(config, p);
        ASSERT_EQ(header.tensors.size(), expected.size());
        for (const auto& [name, e] : expected) {
            const auto it = header.tensors.find(name);
            ASSERT_NE(it, header.tensors.end()) << name;
            EXPECT_EQ(it->second.dtype, e.dtype) << name;
            EXPECT_EQ(it->second.shape, e.shape) << name;
        }

        const auto& entry = header.tensors.at("input_ids");
        std::vector<int32_t> ids(entry.bytes / sizeof(int32_t));
        std::memcpy(ids.data(), file.bytes().data() + entry.offset, entry.bytes);
        const auto prompt = tok.encode(engine::chat_prompt(p.messages, true), false);
        ASSERT_EQ(int(prompt.size()), p.prompt_len);
        EXPECT_EQ(std::vector<int32_t>(ids.begin(), ids.begin() + p.prompt_len), prompt);
        EXPECT_EQ(std::vector<int32_t>(ids.begin() + p.prompt_len, ids.end()), p.generated_ids);
    }
    EXPECT_EQ(traced, 1);
}
