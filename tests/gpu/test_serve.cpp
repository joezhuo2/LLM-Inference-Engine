#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/draft_model.h"
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
constexpr int kTokenBudgets[] = {kRequests, 23, 40, 100, kContext};

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
                                                int* preemptions = nullptr, bool graphs = false,
                                                int token_budget = 0, int max_draft = 0,
                                                engine::DraftCounts* drafts = nullptr,
                                                engine::DraftModel* draft_model = nullptr) {
        KvCache cache(w().config, num_blocks, kBlockSize);
        PagedRunner runner(w().config, w().weights, cache, context, max_batch, context, stream);
        Scheduler scheduler(SchedulerConfig{.num_blocks = num_blocks,
                                            .block_size = kBlockSize,
                                            .max_batch = max_batch,
                                            .max_prefill_tokens = context,
                                            .max_context = context,
                                            .max_draft = max_draft});
        if (draft_model)
            scheduler.set_draft_source(
                [draft_model](engine::SeqId id, std::span<const int32_t> context, int want) {
                    return draft_model->propose(id, context, want);
                });
        Sampler sampler(max_batch);
        for (const Request& r : reqs) scheduler.submit(r);
        std::map<int64_t, std::vector<int32_t>> out;
        if (preemptions) *preemptions = 0;
        std::optional<engine::DecodeGraphs> decode_graphs;
        if (graphs) decode_graphs.emplace(runner, std::vector<int64_t>{1, 2, 4, 8, max_batch});
        for (const Request& r : engine::serve(scheduler, runner, sampler,
                                              graphs ? &*decode_graphs : nullptr, token_budget)) {
            out[r.id] = r.output;
            if (preemptions) *preemptions += r.preemptions;
        }
        if (drafts) *drafts = scheduler.draft_counts();
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

TEST_F(ServeTest, AStepTraceRebuildsEveryOutputThroughPreemption) {
    constexpr int kBlocks = 56;
    constexpr int kTraceContext = 512;
    KvCache cache(w().config, kBlocks, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, kTraceContext, kRequests, kTraceContext,
                       stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = kBlocks,
                                        .block_size = kBlockSize,
                                        .max_batch = kRequests,
                                        .max_prefill_tokens = kTraceContext,
                                        .max_context = kTraceContext});
    Sampler sampler(kRequests);
    const int free_at_start = scheduler.blocks().num_free();
    for (const Request& r : requests(SamplingParams{0.0f, 0})) scheduler.submit(r);
    engine::DeviceBuffer next(size_t(kRequests) * sizeof(int32_t));
    std::vector<engine::StepTrace> trace;
    std::map<int64_t, std::vector<int32_t>> outputs;
    int preemptions = 0;
    while (scheduler.num_waiting() + scheduler.num_running() > 0) {
        for (const Request& r :
             engine::serve_step(scheduler, runner, sampler, next, nullptr, nullptr, &trace)) {
            outputs[r.id] = r.output;
            preemptions += r.preemptions;
        }
    }
    ASSERT_GT(preemptions, 0);
    ASSERT_EQ(outputs.size(), size_t(kRequests));

    std::map<int64_t, std::vector<int32_t>> rebuilt;
    std::map<int64_t, int> finished;
    for (size_t i = 0; i < trace.size(); ++i) {
        const engine::StepTrace& s = trace[i];
        ASSERT_EQ(s.tokens.size(), s.seqs.size()) << "step " << i;
        EXPECT_LE(s.start, s.end);
        if (i > 0) {
            EXPECT_LE(trace[i - 1].end, s.start);
        }
        for (size_t j = 0; j < s.seqs.size(); ++j) rebuilt[s.seqs[j]].push_back(s.tokens[j]);
        for (const int64_t id : s.finished) ++finished[id];
        EXPECT_GE(s.free_blocks, 0);
        EXPECT_LE(s.free_blocks, kBlocks);
    }
    EXPECT_EQ(rebuilt, outputs);
    for (const auto& [id, count] : finished) EXPECT_EQ(count, 1) << "request " << id;
    EXPECT_EQ(finished.size(), size_t(kRequests));
    EXPECT_TRUE(trace.front().prefill);
    EXPECT_EQ(trace.back().waiting + trace.back().running, 0);
    EXPECT_EQ(trace.back().free_blocks, free_at_start);
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

TEST_F(ServeTest, TokenBudgetsKeepTheGreedyOutputsOfEveryRequestUpToTies) {
    const auto reqs = requests(SamplingParams{0.0f, 0});
    const auto whole = run(reqs, 1024, kRequests);
    for (const int budget : kTokenBudgets) {
        SCOPED_TRACE("token budget " + std::to_string(budget));
        const auto chunked = run(reqs, 1024, kRequests, kContext, nullptr, false, budget);
        ASSERT_EQ(chunked.size(), size_t(kRequests));
        size_t identical = 0;
        for (const Request& r : reqs) {
            SCOPED_TRACE("request " + std::to_string(r.id));
            identical += chunked.at(r.id) == whole.at(r.id);
            expect_equal_up_to_a_tie(r, chunked.at(r.id), whole.at(r.id));
        }
        std::printf("token budget %d: %zu of %d identical to whole-prompt prefill\n", budget,
                    identical, kRequests);
    }
}

TEST_F(ServeTest, TokenBudgetsKeepTheOutputsThroughForcedPreemptionAndGraphsUpToTies) {
    const auto reqs = requests(SamplingParams{0.0f, 0});
    const auto roomy = run(reqs, 1024, kRequests, 512);
    for (const int budget : {kRequests, 40}) {
        for (const bool graphs : {false, true}) {
            SCOPED_TRACE("token budget " + std::to_string(budget) + (graphs ? " graphs" : ""));
            int preemptions = 0;
            const auto tight = run(reqs, 56, kRequests, 512, &preemptions, graphs, budget);
            EXPECT_GT(preemptions, 0);
            ASSERT_EQ(tight.size(), size_t(kRequests));
            size_t identical = 0;
            for (const Request& r : reqs) {
                SCOPED_TRACE("request " + std::to_string(r.id));
                identical += tight.at(r.id) == roomy.at(r.id);
                expect_equal_up_to_a_tie(r, tight.at(r.id), roomy.at(r.id));
            }
            std::printf("token budget %d%s: %d preemptions, %zu of %d identical\n", budget,
                        graphs ? " with graphs" : "", preemptions, identical, kRequests);
        }
    }
}

TEST_F(ServeTest, SpeculationKeepsTheGreedyOutputsOfEveryRequestUpToTies) {
    constexpr int kSpeculating = 4;
    constexpr int kMaxDraft = 8;
    std::vector<Request> reqs = requests(SamplingParams{0.0f, 0});
    reqs.resize(kSpeculating);
    for (Request& r : reqs) {
        const auto half = long(r.prompt.size() / 2);
        r.prompt.insert(r.prompt.end(), r.prompt.begin() + 1, r.prompt.begin() + 1 + half);
    }
    const auto plain = run(reqs, 1024, kRequests);
    struct Case {
        bool graphs;
        int token_budget;
    };
    for (const Case c : {Case{false, 0}, Case{true, 0}, Case{false, 40}, Case{true, 40}}) {
        SCOPED_TRACE(std::string(c.graphs ? "graphs" : "eager") + ", token budget " +
                     std::to_string(c.token_budget));
        engine::DraftCounts drafts;
        const auto speculative = run(reqs, 1024, kRequests, kContext, nullptr, c.graphs,
                                     c.token_budget, kMaxDraft, &drafts);
        ASSERT_EQ(speculative.size(), reqs.size());
        size_t identical = 0;
        for (const Request& r : reqs) {
            SCOPED_TRACE("request " + std::to_string(r.id));
            identical += speculative.at(r.id) == plain.at(r.id);
            expect_equal_up_to_a_tie(r, speculative.at(r.id), plain.at(r.id));
        }
        std::printf(
            "%s, token budget %d: %lld of %lld draft tokens accepted, %zu of %d identical\n",
            c.graphs ? "graphs" : "eager", c.token_budget, static_cast<long long>(drafts.accepted),
            static_cast<long long>(drafts.proposed), identical, kSpeculating);
        EXPECT_GT(drafts.proposed, 0);
        EXPECT_GT(drafts.accepted, 0);
    }
}

TEST_F(ServeTest, TheModelAsItsOwnDraftModelAgreesWithItselfAndKeepsTheOutputs) {
    constexpr int kSpeculating = 4;
    constexpr int kMaxDraft = 4;
    std::vector<Request> reqs = requests(SamplingParams{0.0f, 0});
    reqs.resize(kSpeculating);
    const auto plain = run(reqs, 1024, kRequests);
    for (const int token_budget : {0, 40}) {
        SCOPED_TRACE("token budget " + std::to_string(token_budget));
        engine::DraftModel draft_model(w(), kContext, kSpeculating, stream);
        engine::DraftCounts drafts;
        const auto speculative = run(reqs, 1024, kRequests, kContext, nullptr, false, token_budget,
                                     kMaxDraft, &drafts, &draft_model);
        ASSERT_EQ(speculative.size(), reqs.size());
        for (const Request& r : reqs) {
            SCOPED_TRACE("request " + std::to_string(r.id));
            expect_equal_up_to_a_tie(r, speculative.at(r.id), plain.at(r.id));
        }
        std::printf("token budget %d: %lld of %lld draft tokens accepted, %lld draft forwards\n",
                    token_budget, static_cast<long long>(drafts.accepted),
                    static_cast<long long>(drafts.proposed),
                    static_cast<long long>(draft_model.forwards()));
        EXPECT_GT(drafts.proposed, 0);
        EXPECT_GT(drafts.accepted * 10, drafts.proposed * 8);
    }
}

TEST_F(ServeTest, SpeculationKeepsSampledOutputsUntilNearTies) {
    constexpr int kSpeculating = 4;
    std::vector<Request> reqs = requests(SamplingParams{0.8f, 4321, 40, 0.9f});
    reqs.resize(kSpeculating);
    for (Request& r : reqs) {
        const auto half = long(r.prompt.size() / 2);
        r.prompt.insert(r.prompt.end(), r.prompt.begin() + 1, r.prompt.begin() + 1 + half);
    }
    const auto plain = run(reqs, 1024, kRequests);
    engine::DraftCounts drafts;
    const auto speculative = run(reqs, 1024, kRequests, kContext, nullptr, false, 0, 8, &drafts);
    size_t identical = 0;
    for (const Request& r : reqs) identical += speculative.at(r.id) == plain.at(r.id);
    std::printf("sampled: %lld of %lld draft tokens accepted, %zu of %d identical\n",
                static_cast<long long>(drafts.accepted), static_cast<long long>(drafts.proposed),
                identical, kSpeculating);
    EXPECT_GT(drafts.proposed, 0);
    EXPECT_GE(identical, size_t(kSpeculating - 1));
}

TEST_F(ServeTest, ASpeculativeStepTraceListsEveryKeptToken) {
    constexpr int kTraceContext = 512;
    KvCache cache(w().config, 256, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, kTraceContext, kRequests, kTraceContext,
                       stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = 256,
                                        .block_size = kBlockSize,
                                        .max_batch = kRequests,
                                        .max_prefill_tokens = kTraceContext,
                                        .max_context = kTraceContext,
                                        .max_draft = 8});
    Sampler sampler(kRequests);
    std::vector<Request> reqs = requests(SamplingParams{0.0f, 0});
    reqs.resize(2);
    for (Request& r : reqs) {
        const auto half = long(r.prompt.size() / 2);
        r.prompt.insert(r.prompt.end(), r.prompt.begin() + 1, r.prompt.begin() + 1 + half);
        scheduler.submit(r);
    }
    engine::DeviceBuffer next(size_t(kRequests) * sizeof(int32_t));
    std::vector<engine::StepTrace> trace;
    engine::ServeStats stats;
    std::map<int64_t, std::vector<int32_t>> outputs;
    while (scheduler.num_waiting() + scheduler.num_running() > 0) {
        for (const Request& r :
             engine::serve_step(scheduler, runner, sampler, next, &stats, nullptr, &trace))
            outputs[r.id] = r.output;
    }

    std::map<int64_t, std::vector<int32_t>> rebuilt;
    int64_t drafted = 0, accepted = 0, longest = 0;
    for (const engine::StepTrace& s : trace) {
        ASSERT_EQ(s.tokens.size(), s.seqs.size());
        for (size_t j = 0; j < s.seqs.size(); ++j) rebuilt[s.seqs[j]].push_back(s.tokens[j]);
        EXPECT_LE(s.accepted, s.drafted);
        EXPECT_LE(s.drafted, kRequests - 1);
        drafted += s.drafted;
        accepted += s.accepted;
        longest = std::max<int64_t>(longest, int64_t(s.seqs.size()));
    }
    EXPECT_EQ(rebuilt, outputs);
    EXPECT_EQ(drafted, scheduler.draft_counts().proposed);
    EXPECT_EQ(accepted, scheduler.draft_counts().accepted);
    EXPECT_EQ(stats.drafted, drafted);
    EXPECT_EQ(stats.accepted, accepted);
    EXPECT_GT(accepted, 0);
    EXPECT_GT(longest, 2);
    EXPECT_LT(stats.steps, 1 + kNewTokens);
}
