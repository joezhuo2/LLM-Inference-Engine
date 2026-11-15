#include <gtest/gtest.h>
#include <sys/wait.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include "support/paths.h"

namespace {

struct EngineRun {
    int status = -1;
    std::string out;
};

EngineRun run_engine(const std::string& args) {
    const std::string command = std::string(ENGINE_BINARY) + " generate --model " +
                                engine::test::model_dir().string() + " " + args + " 2>&1";
    EngineRun r;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return r;
    char buf[4096];
    while (const size_t n = fread(buf, 1, sizeof(buf), pipe)) r.out.append(buf, n);
    const int status = pclose(pipe);
    r.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return r;
}

std::string answer(const std::string& out) {
    return out.substr(0, out.find("prompt tokens"));
}

class GenerateCommandTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto file = engine::test::model_dir() / "model.safetensors";
        if (!std::filesystem::exists(file)) GTEST_SKIP() << file << " not found";
    }
};

}  // namespace

TEST_F(GenerateCommandTest, TensorParallelGivesTheSameGreedyText) {
    const std::string args = "--prompt 'Name three colors.' --max-new-tokens 24";
    const EngineRun whole = run_engine(args);
    const EngineRun sharded = run_engine(args + " --tensor-parallel 2");
    ASSERT_EQ(whole.status, 0) << whole.out;
    ASSERT_EQ(sharded.status, 0) << sharded.out;
    EXPECT_FALSE(answer(whole.out).empty());
    EXPECT_EQ(answer(sharded.out), answer(whole.out));
}

TEST_F(GenerateCommandTest, TensorParallelIsRejectedWithSpeculate) {
    const EngineRun r = run_engine("--prompt hi --tensor-parallel 2 --speculate 4");
    EXPECT_NE(r.status, 0);
    EXPECT_NE(r.out.find("--tensor-parallel cannot be combined with --speculate"),
              std::string::npos)
        << r.out;
}
