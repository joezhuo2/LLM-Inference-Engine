#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
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
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    std::optional<uint64_t> seed;
};

#if ENGINE_HAS_CUDA
void run(const Options& o) {
    using Clock = std::chrono::steady_clock;
    std::random_device entropy;
    const SamplingParams sampling{
        o.temperature, o.seed.value_or(uint64_t(entropy()) << 32 | entropy()), o.top_k, o.top_p};
    validate(sampling);

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
    const auto out = generate(runner, prompt, int(context - prompt_len), w.config.eos_id, sampling,
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
    if (sampling.temperature > 0)
        std::fprintf(stderr, "temperature %g, top-k %d, top-p %g, seed %llu\n",
                     double(sampling.temperature), sampling.top_k, double(sampling.top_p),
                     static_cast<unsigned long long>(sampling.seed));
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
    CLI::App* cmd = app.add_subcommand("generate", "Answer one chat message");
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
    cmd->add_option("--temperature", opts->temperature,
                    "Sample from softmax(logits / temperature); 0 decodes greedily")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-k", opts->top_k,
                    "When sampling, draw only from the k most likely tokens and ties with the "
                    "k-th; 0 keeps every token")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-p", opts->top_p,
                    "When sampling, draw only from the most likely tokens whose probability "
                    "reaches p, in (0, 1]; 1 keeps every token")
        ->capture_default_str()
        ->check(CLI::Range(0.0f, 1.0f));
    cmd->add_option("--seed", opts->seed,
                    "Seed for sampling; the same seed repeats the answer (default: random, "
                    "printed)");
    cmd->callback([opts] { run(*opts); });
}

}  // namespace engine::app
