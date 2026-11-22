#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "engine/runtime/device_buffer.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/runtime/serve.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/model.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::Request;
using engine::Scheduler;
using engine::SchedulerConfig;

namespace {

constexpr int kBlockSize = 16;
constexpr int kContext = 512;
constexpr int kNewTokens = 32;
constexpr int kTieUlps = 3;

using Tokens = std::vector<int32_t>;

struct Options {
    int num_blocks = 256;
    int max_batch = 8;
    int token_budget = 0;
    int max_draft = 0;
    bool flash = true;
    bool stagger = true;
};

struct Rig {
    Rig(const engine::DeviceWeights& w, const Options& o, bool cache, cudaStream_t stream)
        : kv(w.config, o.num_blocks, kBlockSize),
          runner(w.config, w.weights, kv, kContext, o.max_batch, kContext, stream),
          scheduler(SchedulerConfig{.num_blocks = o.num_blocks,
                                    .block_size = kBlockSize,
                                    .max_batch = o.max_batch,
                                    .max_prefill_tokens = kContext,
                                    .max_context = kContext,
                                    .max_draft = o.max_draft,
                                    .prefix_cache = cache}),
          sampler(o.max_batch),
          next(size_t(o.max_batch) * sizeof(int32_t)),
          token_budget(o.token_budget) {
        runner.set_flash_prefill(o.flash);
    }

    std::vector<Request> step() {
        return engine::serve_step(scheduler, runner, sampler, next, nullptr, nullptr, nullptr,
                                  token_budget);
    }

    engine::KvCache kv;
    engine::PagedRunner runner;
    Scheduler scheduler;
    engine::Sampler sampler;
    engine::DeviceBuffer next;
    int token_budget;
};

struct Result {
    std::map<int64_t, Tokens> outputs;
    engine::PrefixCounts counts;
    int preemptions = 0;
};

class PrefixCacheServeTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto why = engine::test::reference_skip_reason(); !why.empty())
            GTEST_SKIP() << why;
        for (const auto& prompt : engine::test::load_reference_manifest()) {
            const engine::test::SafetensorsFile ref(prompt.file);
            const auto ids = ref.read<int32_t>("input_ids");
            prompts.emplace_back(ids.begin(), ids.begin() + prompt.prompt_len);
        }
        ASSERT_GE(prompts.size(), 14u);
        system = prompts[0];
        for (size_t i = 1; i < 6; ++i)
            system.insert(system.end(), prompts[i].begin() + 1, prompts[i].end());
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }

    Tokens with_system(size_t suffix) const {
        Tokens prompt = system;
        prompt.insert(prompt.end(), prompts[suffix].begin() + 1, prompts[suffix].end());
        return prompt;
    }

    std::vector<Request> shared_system_requests() const {
        std::vector<Request> out;
        for (int i = 0; i < 8; ++i)
            out.push_back(Request{.id = i,
                                  .prompt = with_system(i < 6 ? size_t(6 + i) : 7),
                                  .max_new_tokens = kNewTokens});
        return out;
    }

    Result run(const std::vector<Request>& reqs, const Options& o, bool cache) {
        Rig rig(w(), o, cache, stream);
        Result out;
        size_t submitted = 0;
        while (out.outputs.size() < reqs.size()) {
            do {
                if (submitted < reqs.size()) rig.scheduler.submit(reqs[submitted++]);
            } while (!o.stagger && submitted < reqs.size());
            for (const Request& r : rig.step()) {
                out.outputs[r.id] = r.output;
                out.preemptions += r.preemptions;
            }
        }
        out.counts = rig.scheduler.prefix_counts();
        return out;
    }

    void expect_equal_up_to_a_tie(const Request& r, const Tokens& got, const Tokens& want) {
        ASSERT_EQ(got.size(), want.size());
        const size_t p =
            size_t(std::mismatch(got.begin(), got.end(), want.begin()).first - got.begin());
        if (p == got.size()) return;
        engine::ModelRunner oracle(w().config, w().weights, kContext, kContext, stream);
        Tokens ids = r.prompt;
        ids.insert(ids.end(), want.begin(), want.begin() + long(p));
        const engine::Tensor logits = oracle.forward(ids);
        std::vector<uint16_t> host(size_t(w().config.vocab));
        ASSERT_EQ(cudaMemcpy(host.data(), logits.data, host.size() * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        float top = engine::test::from_bf16(host[0]);
        for (const uint16_t x : host) top = std::max(top, engine::test::from_bf16(x));
        const float slack = float(kTieUlps) * std::ldexp(1.0f, std::ilogb(top) - 7);
        for (const int32_t token : {got[p], want[p]})
            EXPECT_GE(engine::test::from_bf16(host[size_t(token)]), top - slack)
                << "request " << r.id << " differs at token " << p << " but token " << token
                << " is not within " << kTieUlps << " BF16 ulps of the best logit " << top;
    }

    Result expect_same_outputs(const std::vector<Request>& reqs, const Options& o) {
        const Result off = run(reqs, o, false);
        const Result on = run(reqs, o, true);
        EXPECT_EQ(off.counts.cached, 0);
        EXPECT_GT(on.counts.cached, 0);
        int identical = 0;
        for (const Request& r : reqs) {
            identical += on.outputs.at(r.id) == off.outputs.at(r.id);
            expect_equal_up_to_a_tie(r, on.outputs.at(r.id), off.outputs.at(r.id));
        }
        std::printf(
            "prefix cache: %lld of %lld prefill tokens cached, %d preemptions, %d of %zu "
            "outputs identical\n",
            static_cast<long long>(on.counts.cached), static_cast<long long>(on.counts.prefill),
            on.preemptions, identical, reqs.size());
        return on;
    }

    std::vector<uint16_t> keys_and_values(engine::KvCache& kv, std::span<const int> blocks) const {
        std::vector<uint16_t> out;
        for (int layer = 0; layer < w().config.layers; ++layer) {
            for (const engine::Tensor& t : {kv.k(layer), kv.v(layer)}) {
                const int64_t per_block = t.numel() / t.shape[0];
                for (const int block : blocks) {
                    const size_t at = out.size();
                    out.resize(at + size_t(per_block));
                    EXPECT_EQ(
                        cudaMemcpy(out.data() + at,
                                   static_cast<const uint16_t*>(t.data) + block * per_block,
                                   size_t(per_block) * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                        cudaSuccess);
                }
            }
        }
        return out;
    }

    std::vector<Tokens> prompts;
    Tokens system;
    engine::test::CudaStream stream;
};

TEST_F(PrefixCacheServeTest, AHitReadsTheKeysAndValuesItsOwnPrefillWouldWrite) {
    const Request first{.id = 0, .prompt = with_system(6), .max_new_tokens = kNewTokens};
    const Request second{.id = 1, .prompt = with_system(11), .max_new_tokens = kNewTokens};
    const size_t shared = system.size() / kBlockSize;
    for (const bool flash : {true, false}) {
        SCOPED_TRACE(flash ? "flash prefill" : "paged prefill");
        Rig on(w(), {.flash = flash}, true, stream);
        on.scheduler.submit(first);
        on.step();
        on.scheduler.submit(second);
        on.step();
        ASSERT_EQ(on.scheduler.request(1).output.size(), 1u);
        EXPECT_EQ(on.scheduler.prefix_counts().cached, int64_t(shared) * kBlockSize);
        const std::span<const int> reused = std::span(on.scheduler.blocks().table(1)).first(shared);
        EXPECT_TRUE(
            std::ranges::equal(reused, std::span(on.scheduler.blocks().table(0)).first(shared)));

        Rig off(w(), {.flash = flash}, false, stream);
        off.scheduler.submit(second);
        off.step();
        EXPECT_TRUE(
            keys_and_values(on.kv, reused) ==
            keys_and_values(off.kv, std::span(off.scheduler.blocks().table(1)).first(shared)));
    }
}

TEST_F(PrefixCacheServeTest, ASharedSystemPromptAndRepeatsKeepTheGreedyOutputs) {
    const Result on = expect_same_outputs(shared_system_requests(), {});
    EXPECT_GE(on.counts.cached, int64_t(7 * (system.size() / kBlockSize) * kBlockSize));
}

TEST_F(PrefixCacheServeTest, ConcurrentRequestsSharingAPrefixKeepTheGreedyOutputs) {
    expect_same_outputs(shared_system_requests(), {.max_batch = 4, .stagger = false});
}

TEST_F(PrefixCacheServeTest, EvictionInASmallPoolKeepsTheGreedyOutputs) {
    const Result on = expect_same_outputs(shared_system_requests(), {.num_blocks = 40});
    EXPECT_GT(on.preemptions, 0);
}

TEST_F(PrefixCacheServeTest, SpeculationKeepsTheGreedyOutputs) {
    expect_same_outputs(shared_system_requests(), {.max_draft = 4});
}

TEST_F(PrefixCacheServeTest, ChunkedAndPagedPrefillKeepTheGreedyOutputs) {
    expect_same_outputs(shared_system_requests(), {.token_budget = 64});
    expect_same_outputs(shared_system_requests(), {.flash = false});
    expect_same_outputs(shared_system_requests(), {.token_budget = 40, .flash = false});
}

TEST_F(PrefixCacheServeTest, ASecondTurnReusesTheGeneratedTokensOfTheFirst) {
    const Request first{.id = 0, .prompt = with_system(6), .max_new_tokens = kNewTokens};
    const Tokens answer = run({first}, {.max_batch = 1}, false).outputs.at(0);
    Tokens turn = first.prompt;
    turn.insert(turn.end(), answer.begin(), answer.end());
    turn.insert(turn.end(), prompts[7].begin() + 1, prompts[7].end());
    const Request second{.id = 1, .prompt = turn, .max_new_tokens = kNewTokens};
    const Result on = expect_same_outputs({first, second}, {.max_batch = 1});
    EXPECT_EQ(on.outputs.at(0), answer);
    EXPECT_GT(on.counts.cached, int64_t(first.prompt.size()));
}

}  // namespace
