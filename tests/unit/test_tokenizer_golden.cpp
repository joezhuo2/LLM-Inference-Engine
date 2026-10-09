#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "engine/tokenizer/tokenizer.h"
#include "support/paths.h"

TEST(TokenizerGolden, MatchesHuggingFace) {
    const auto model = engine::test::model_dir() / "tokenizer.model";
    const auto golden = engine::test::golden_dir() / "tokenizer.json";
    if (!std::filesystem::exists(model)) GTEST_SKIP() << model << " not found";
    if (!std::filesystem::exists(golden))
        GTEST_SKIP() << golden << " not found, run scripts/tokenizer_golden.py";

    const engine::Tokenizer tok(model);
    const auto cases = nlohmann::json::parse(std::ifstream(golden)).at("cases");
    ASSERT_FALSE(cases.empty());
    int failures = 0;
    for (const auto& c : cases) {
        const auto text = c.at("text").get<std::string>();
        const auto ids = c.at("ids").get<std::vector<int32_t>>();
        const auto encoded = tok.encode(text, true);
        const auto decoded = tok.decode(ids);
        if (encoded == ids && decoded == c.at("decoded").get<std::string>()) continue;
        if (++failures <= 10) {
            ADD_FAILURE() << "text " << nlohmann::json(text).dump() << ": encoded "
                          << nlohmann::json(encoded).dump() << ", expected " << c.at("ids").dump()
                          << "; decoded " << nlohmann::json(decoded).dump() << ", expected "
                          << c.at("decoded").dump();
        }
    }
    EXPECT_EQ(failures, 0) << "of " << cases.size() << " cases";
}
