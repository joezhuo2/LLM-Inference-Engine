#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <vector>

#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/generate.h"
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

using engine::KvCache;
using engine::PagedRunner;
using engine::Request;
using engine::Sampler;
using engine::SamplingParams;
using engine::Scheduler;
using engine::SchedulerConfig;

namespace {

constexpr int kRequests = 16;
constexpr int kNewTokens = 48;
constexpr int kBlockSize = 16;
constexpr int kContext = 2048;
constexpr size_t kMinIdenticalSampled = 10;
constexpr size_t kMinIdenticalPreempted = 14;
constexpr int kTieUlps = 2;

class ServeTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto why = engine::test::reference_skip_reason(); !why.empty())
            GTEST_SKIP() << why;
        for (const auto& prompt : engine::test::load_reference_manifest()) {
            if (int(prompts.size()) == kRequests) break;
            const engine::test::SafetensorsFile ref(prompt.file);
            const auto ids = ref.read<int32_t>("input_ids");
            prompts.emplace_back(ids.begin(), ids.begin() + prompt.prompt_len);
        }
        ASSERT_EQ(int(prompts.size()), kRequests);
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }

    std::vector<Request> requests(const SamplingParams& base) const {
        std::vector<Request> out;
        for (int i = 0; i < kRequests; ++i) {
            Request r{.id = i, .max_new_tokens = kNewTokens, .params = base};
            r.params.seed = base.seed + uint64_t(i);
            r.prompt = prompts[size_t(i)];
            out.push_back(r);
        }
        return out;
    }

    std::map<int64_t, std::vector<int32_t>> run(const std::vector<Request>& reqs, int num_blocks,
                                                int max_batch, int context = kContext,
                                                int* preemptions = nullptr, bool graphs = false) {
        KvCache cache(w().config, num_blocks, kBlockSize);
        PagedRunner runner(w().config, w().weights, cache, context, max_batch, context, stream);
        Scheduler scheduler(SchedulerConfig{.num_blocks = num_blocks,
                                            .block_size = kBlockSize,
                                            .max_batch = max_batch,
                                            .max_prefill_tokens = context,
                                            .max_context = context});
        Sampler sampler(max_batch);
        for (const Request& r : reqs) scheduler.submit(r);
        std::map<int64_t, std::vector<int32_t>> out;
        if (preemptions) *preemptions = 0;
        std::optional<engine::DecodeGraphs> decode_graphs;
        if (graphs) decode_graphs.emplace(runner, std::vector<int64_t>{1, 2, 4, 8, max_batch});
        for (const Request& r :
             engine::serve(scheduler, runner, sampler, graphs ? &*decode_graphs : nullptr)) {
            out[r.id] = r.output;
            if (preemptions) *preemptions += r.preemptions;
        }
        return out;
    }

    void expect_equal_up_to_a_tie(const Request& r, const std::vector<int32_t>& got,
                                  const std::vector<int32_t>& want) {
        ASSERT_EQ(got.size(), want.size());
        const size_t p =
            size_t(std::mismatch(got.begin(), got.end(), want.begin()).first - got.begin());
        if (p == got.size()) return;
        engine::ModelRunner oracle(w().config, w().weights, 512, 512, stream);
        std::vector<int32_t> ids = r.prompt;
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
                << "the outputs differ at token " << p << " but token " << token
                << " is not within " << kTieUlps << " BF16 ulps of the best logit " << top;
    }

    std::vector<std::vector<int32_t>> prompts;
    engine::test::CudaStream stream;
};

}  // namespace

TEST_F(ServeTest, SixteenConcurrentGreedyRequestsMatchSixteenSoloRunsUpToTies) {
    const auto reqs = requests(SamplingParams{0.0f, 0});
    const auto together = run(reqs, 1024, kRequests);
    ASSERT_EQ(together.size(), size_t(kRequests));
    for (const Request& r : reqs) {
        SCOPED_TRACE("request " + std::to_string(r.id));
        EXPECT_EQ(together.at(r.id).size(), size_t(kNewTokens));
        expect_equal_up_to_a_tie(r, together.at(r.id), run({r}, 256, 1).at(r.id));
    }
}

TEST_F(ServeTest, SixteenConcurrentSampledRequestsAgreeWithSoloRunsUntilNearTies) {
    const auto reqs = requests(SamplingParams{0.8f, 1234, 40, 0.9f});
    const auto together = run(reqs, 1024, kRequests);
    ASSERT_EQ(together.size(), size_t(kRequests));
    size_t identical = 0, shortest_prefix = kNewTokens;
    for (const Request& r : reqs) {
        const auto& a = together.at(r.id);
        const auto b = run({r}, 256, 1).at(r.id);
        ASSERT_EQ(a.size(), b.size());
        const size_t prefix =
            size_t(std::mismatch(a.begin(), a.end(), b.begin()).first - a.begin());
        identical += prefix == a.size();
        shortest_prefix = std::min(shortest_prefix, prefix);
    }
    std::printf("%zu of %d identical to a solo run, shortest common prefix %zu of %d tokens\n",
                identical, kRequests, shortest_prefix, kNewTokens);
    EXPECT_GE(identical, kMinIdenticalSampled);
}

TEST_F(ServeTest, ForcedPreemptionKeepsTheOutputsOfEveryRequest) {
    for (const SamplingParams& params :
         {SamplingParams{0.0f, 0}, SamplingParams{0.8f, 99, 40, 0.9f}}) {
        const auto reqs = requests(params);
        const auto roomy = run(reqs, 1024, kRequests, 512);
        int preemptions = 0;
        const auto tight = run(reqs, 56, kRequests, 512, &preemptions);
        EXPECT_GT(preemptions, 0);
        ASSERT_EQ(tight.size(), size_t(kRequests));
        size_t identical = 0;
        for (const Request& r : reqs) {
            SCOPED_TRACE("request " + std::to_string(r.id));
            const bool same = tight.at(r.id) == roomy.at(r.id);
            identical += same;
            if (params.temperature == 0.0f) {
                EXPECT_TRUE(same);
            }
        }
        if (params.temperature > 0.0f) {
            EXPECT_GE(identical, kMinIdenticalPreempted);
        }
        std::printf("temperature %.1f: %d preemptions, %zu of %d identical\n", params.temperature,
                    preemptions, identical, kRequests);
    }
}

TEST_F(ServeTest, DecodeGraphsKeepTheGreedyOutputsOfEveryRequest) {
    const auto reqs = requests(SamplingParams{0.0f, 0});
    const auto eager = run(reqs, 1024, kRequests);
    const auto graphed = run(reqs, 1024, kRequests, kContext, nullptr, true);
    ASSERT_EQ(graphed.size(), size_t(kRequests));
    for (const Request& r : reqs) {
        SCOPED_TRACE("request " + std::to_string(r.id));
        EXPECT_EQ(graphed.at(r.id), eager.at(r.id));
    }
}

TEST_F(ServeTest, DecodeGraphsSurviveForcedPreemption) {
    const auto reqs = requests(SamplingParams{0.0f, 0});
    const auto roomy = run(reqs, 1024, kRequests, 512);
    int preemptions = 0;
    const auto tight = run(reqs, 56, kRequests, 512, &preemptions, true);
    EXPECT_GT(preemptions, 0);
    ASSERT_EQ(tight.size(), size_t(kRequests));
    for (const Request& r : reqs) {
        SCOPED_TRACE("request " + std::to_string(r.id));
        EXPECT_EQ(tight.at(r.id), roomy.at(r.id));
    }
}
