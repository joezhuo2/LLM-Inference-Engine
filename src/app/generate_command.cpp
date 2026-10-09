#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.h"
#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/text_stream.h"
#include "engine/tokenizer/tokenizer.h"

#if ENGINE_HAS_CUDA
#include "engine/runtime/generate.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/weights.h"
#endif

namespace engine::app {

namespace {

struct Options {
    std::filesystem::path model;
    std::string prompt;
    std::string system;
    int max_new_tokens = 256;
};

#if ENGINE_HAS_CUDA
void run(const Options& o) {
    using Clock = std::chrono::steady_clock;
    const Tokenizer tokenizer(o.model / "tokenizer.model");
    std::vector<ChatMessage> messages;
    if (!o.system.empty()) messages.push_back({"system", o.system});
    messages.push_back({"user", o.prompt});
    const auto prompt = tokenizer.encode(chat_prompt(messages, true), false);

    const DeviceWeights w = load_weights(o.model, nullptr);
    const auto prompt_len = int64_t(prompt.size());
    const int64_t context =
        std::min<int64_t>(prompt_len + o.max_new_tokens, w.config.max_positions);
    if (prompt_len >= context)
        throw std::runtime_error("the prompt has " + std::to_string(prompt_len) +
                                 " tokens; the context is " +
                                 std::to_string(w.config.max_positions));
    ModelRunner runner(w.config, w.weights, prompt_len, context, nullptr);

    TextStream text(tokenizer);
    const auto start = Clock::now();
    Clock::time_point first;
    const auto out = generate_greedy(runner, prompt, int(context - prompt_len), w.config.eos_id,
                                     [&](int32_t token) {
                                         if (first == Clock::time_point{}) first = Clock::now();
                                         const std::string piece = text.push(token);
                                         std::fwrite(piece.data(), 1, piece.size(), stdout);
                                         std::fflush(stdout);
                                     });
    const auto end = Clock::now();
    const double prefill = std::chrono::duration<double>(first - start).count();
    const double decode = std::chrono::duration<double>(end - first).count();
    std::printf("\n");
    std::fflush(stdout);
    std::fprintf(stderr,
                 "%lld prompt tokens, %zu generated; first token after %.1f ms, then %.1f "
                 "tokens/s\n",
                 static_cast<long long>(prompt_len), out.size(), prefill * 1e3,
                 out.size() > 1 ? double(out.size() - 1) / decode : 0.0);
}
#else
void run(const Options&) {
    throw std::runtime_error("generate needs a CUDA build");
}
#endif

}  // namespace

void add_generate_command(CLI::App& app) {
    auto opts = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand("generate", "Answer one chat message with greedy decoding");
    cmd->add_option("--model", opts->model,
                    "Model directory with config.json, model.safetensors and tokenizer.model")
        ->required()
        ->check(CLI::ExistingDirectory);
    cmd->add_option("--prompt", opts->prompt, "The user message")->required();
    cmd->add_option("--system", opts->system, "An optional system message");
    cmd->add_option("--max-new-tokens", opts->max_new_tokens,
                    "Stop after this many tokens unless EOS comes first")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    cmd->callback([opts] { run(*opts); });
}

}  // namespace engine::app
