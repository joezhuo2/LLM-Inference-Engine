#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/runtime/generate.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/sharded_runner.h"
#include "engine/runtime/weights.h"
#include "engine/tokenizer/tokenizer.h"
#include "kernels/cuda_check.h"

using namespace engine;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxTokens = 128;
constexpr int kContext = 256;
constexpr int kNewTokens = 64;

const char* const kPrompts[] = {
    "The old lighthouse keeper climbed the spiral stairs every evening to light the lamp. One "
    "night a storm rolled in from the sea, and he saw a small boat struggling against the waves.",
    "Explain in simple words why the sky is blue during the day and red at sunset.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
};

double used_mib() {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return double(total - free) / (1 << 20);
}

float bf16(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<uint16_t> host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

double max_abs_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, double(std::abs(bf16(a[i]) - bf16(b[i]))));
    return worst;
}

int32_t argmax(const std::vector<uint16_t>& row) {
    return int32_t(std::max_element(row.begin(), row.end(),
                                    [](uint16_t a, uint16_t b) { return bf16(a) < bf16(b); }) -
                   row.begin());
}

template <typename Runner>
std::vector<double> decode_ms(Runner& runner, std::span<const int32_t> prompt, int steps,
                              cudaStream_t stream) {
    runner.reset();
    int32_t token = argmax(host(runner.forward(prompt)));
    std::vector<double> ms;
    for (int i = 0; i < steps; ++i) {
        const auto start = Clock::now();
        const Tensor logits = runner.forward(std::span(&token, 1));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        token = argmax(host(logits));
    }
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "models/tinyllama";
    const int ranks = argc > 2 ? std::atoi(argv[2]) : 0;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const Tokenizer tokenizer(model + "/tokenizer.model");
    std::vector<std::vector<int32_t>> prompts;
    for (const char* p : kPrompts) prompts.push_back(tokenizer.encode(p, true));

    const double before = used_mib();
    std::vector<DeviceWeights> owned;
    std::vector<ModelWeights> shards;
    for (int r = 0; r < std::max(ranks, 1); ++r) {
        owned.push_back(ranks == 0 ? load_weights(model, stream)
                                   : load_shard_weights(model, r, ranks, stream));
        shards.push_back(owned.back().weights);
    }
    const ModelConfig config = load_config(model + "/config.json");
    std::unique_ptr<ModelRunner> whole_runner;
    std::unique_ptr<ShardedRunner> sharded_runner;
    if (ranks == 0)
        whole_runner =
            std::make_unique<ModelRunner>(config, shards[0], kMaxTokens, kContext, stream);
    else
        sharded_runner =
            std::make_unique<ShardedRunner>(config, shards, kMaxTokens, kContext, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double mib = used_mib() - before;

    std::vector<double> ms;
    for (const auto& p : prompts) {
        const auto run = ranks == 0 ? decode_ms(*whole_runner, p, kNewTokens, stream)
                                    : decode_ms(*sharded_runner, p, kNewTokens, stream);
        ms.insert(ms.end(), run.begin() + 1, run.end());
    }
    std::sort(ms.begin(), ms.end());
    const std::string label = ranks == 0
                                  ? "whole model, ModelRunner"
                                  : std::to_string(ranks) + (ranks == 1 ? " rank" : " ranks");
    if (ranks == 0) {
        std::printf("| %s | %.1f | reference | reference | %.2f | %.2f to %.2f |\n", label.c_str(),
                    mib, ms[ms.size() / 2], ms.front(), ms.back());
        return 0;
    }

    const DeviceWeights whole_weights = load_weights(model, stream);
    ModelRunner whole(config, whole_weights.weights, kMaxTokens, kContext, stream);
    double worst = 0.0;
    int equal = 0, total = 0;
    for (const auto& p : prompts) {
        whole.reset();
        sharded_runner->reset();
        const auto e = host(whole.forward(p, true));
        worst = std::max(worst, max_abs_diff(host(sharded_runner->forward(p, true)), e));
        int32_t token = argmax(std::vector<uint16_t>(e.end() - config.vocab, e.end()));
        for (int i = 0; i < kNewTokens; ++i) {
            const auto er = host(whole.forward(std::span(&token, 1)));
            worst = std::max(worst,
                             max_abs_diff(host(sharded_runner->forward(std::span(&token, 1))), er));
            token = argmax(er);
        }
        const auto a = generate(*sharded_runner, p, kNewTokens, -1);
        const auto b = generate(whole, p, kNewTokens, -1);
        size_t same = 0;
        while (same < a.size() && a[same] == b[same]) ++same;
        equal += int(same);
        total += int(b.size());
    }
    std::printf("| %s | %.1f | %.4f | %d of %d | %.2f | %.2f to %.2f |\n", label.c_str(), mib,
                worst, equal, total, ms[ms.size() / 2], ms.front(), ms.back());
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
