#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/text_stream.h"
#include "engine/tokenizer/tokenizer.h"

using namespace engine;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kWarmup = 200;
constexpr int kRuns = 2001;

template <typename F>
double time_us(F&& f) {
    const auto start = Clock::now();
    f();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

template <typename F>
void row(const char* stage, size_t tokens, F&& f) {
    for (int i = 0; i < kWarmup; ++i) f();
    std::vector<double> us(kRuns);
    for (double& v : us) v = time_us(f);
    std::sort(us.begin(), us.end());
    std::printf("| %s | %zu | %.2f | %.2f to %.2f |\n", stage, tokens, us[kRuns / 2],
                us[kRuns / 20], us[kRuns - 1 - kRuns / 20]);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "models/tinyllama";
    const std::string user = argc > 2 ? argv[2] : "What is the capital of Canada?";
    const std::string answer = argc > 3 ? argv[3] : "The capital of Canada is Ottawa.";
    const Tokenizer tokenizer(model + "/tokenizer.model");
    const std::vector<ChatMessage> messages{{"user", user}};
    const std::string text = chat_prompt(messages, true);
    const std::vector<int32_t> prompt = tokenizer.encode(text, false);
    const std::vector<int32_t> reply = tokenizer.encode(answer, false);

    std::printf("prompt ids:");
    for (int32_t id : prompt) std::printf(" %d", id);
    std::printf("\nanswer ids:");
    for (int32_t id : reply) std::printf(" %d", id);
    std::printf(
        "\n\n| stage | tokens | median us | 5th to 95th percentile |\n| --- | --- | --- | --- |\n");

    size_t sink = 0;
    row("chat template", prompt.size(), [&] { sink += chat_prompt(messages, true).size(); });
    row("encode", prompt.size(), [&] { sink += tokenizer.encode(text, false).size(); });
    row("decode, whole answer", reply.size(), [&] {
        TextStream stream(tokenizer);
        for (int32_t id : reply) sink += stream.push(id).size();
    });
    {
        TextStream stream(tokenizer);
        for (size_t i = 0; i + 1 < reply.size(); ++i) sink += stream.push(reply[i]).size();
        row("decode, last token of the answer", 1, [&] {
            TextStream copy = stream;
            sink += copy.push(reply.back()).size();
        });
    }
    return sink == 0;
}
