#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/sched/paged_batch.h"
#include "engine/sched/scheduler.h"
#include "engine/sched/speculative.h"

using engine::ChunkedStep;
using engine::PagedBatch;
using engine::Request;
using engine::SampleRow;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;
using engine::Step;

namespace {

constexpr int32_t kEos = 2;
constexpr int kBlock = 16;
constexpr int kContext = 256;
constexpr int kMaxBlocks = kContext / kBlock;

using Tokens = std::vector<int32_t>;

SchedulerConfig config(int num_blocks, int max_batch, int max_draft, int32_t eos_id = -1) {
    return {.num_blocks = num_blocks,
            .block_size = kBlock,
            .max_batch = max_batch,
            .max_prefill_tokens = kContext,
            .max_context = kContext,
            .eos_id = eos_id,
            .max_draft = max_draft};
}

Request make(SeqId id, Tokens prompt, int max_new_tokens, uint64_t seed = 0) {
    return Request{.id = id,
                   .prompt = std::move(prompt),
                   .max_new_tokens = max_new_tokens,
                   .params = {0.7f, seed}};
}

Tokens repeating(int length) {
    const Tokens phrase{11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    Tokens out;
    for (int i = 0; i < length; ++i) out.push_back(phrase[size_t(i) % phrase.size()]);
    return out;
}

Tokens random_prompt(std::mt19937& rng, int length) {
    Tokens out;
    for (int i = 0; i < length; ++i)
        out.push_back(3 + int32_t(std::uniform_int_distribution<int>(0, 11)(rng)));
    return out;
}

int32_t model_next(std::span<const int32_t> context, uint64_t seed, uint64_t step) {
    uint64_t h = 1469598103934665603ull ^ seed;
    const auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    for (const int32_t t : context) mix(uint64_t(t));
    mix(step + 0x9e37);
    h ^= h >> 29;
    if (h % 47 == 0) return kEos;
    if (h % 5 != 0)
        for (size_t i = context.size() - 1; i-- > 0;)
            if (context[i] == context.back()) return context[i + 1];
    return int32_t(3 + h % 12);
}

Tokens solo(const Request& r, int32_t eos_id) {
    Tokens context = r.prompt;
    Tokens out;
    while (true) {
        const int32_t t = model_next(context, r.params.seed, out.size());
        out.push_back(t);
        if (t == eos_id || int(out.size()) == r.max_new_tokens) return out;
        context.push_back(t);
    }
}

class HostModel {
public:
    explicit HostModel(const SchedulerConfig& c)
        : cache_(size_t(c.num_blocks) * size_t(c.block_size), -1) {}

    Tokens run(const PagedBatch& batch, std::span<const SampleRow> rows) {
        for (size_t row = 0; row < batch.token_ids.size(); ++row)
            cache_[size_t(batch.slot_mapping[row])] = batch.token_ids[row];
        Tokens out;
        for (size_t i = 0; i < batch.logit_rows.size(); ++i) {
            const auto row = size_t(batch.logit_rows[i]);
            Tokens context;
            for (int pos = 0; pos < batch.context_lens[row]; ++pos) {
                const int32_t block =
                    batch.block_tables[row * size_t(batch.max_blocks) + size_t(pos / kBlock)];
                context.push_back(cache_[size_t(block) * kBlock + size_t(pos % kBlock)]);
            }
            out.push_back(model_next(context, rows[i].params.seed, rows[i].step));
        }
        return out;
    }

private:
    Tokens cache_;
};

struct Served {
    std::map<SeqId, Tokens> done{};
    int steps = 0;
    int preemptions = 0;
    engine::DraftCounts drafts{};
};

void serve(const SchedulerConfig& c, const std::vector<Request>& requests, int budget, Served& out,
           engine::DraftSource source = {}) {
    Scheduler s(c);
    s.set_draft_source(std::move(source));
    HostModel model(c);
    for (const Request& r : requests) s.submit(r);
    while (s.num_waiting() + s.num_running() > 0) {
        ASSERT_LT(out.steps, 200000);
        PagedBatch batch;
        std::vector<SeqId> sampled;
        std::vector<Tokens> drafts;
        if (budget > 0) {
            const ChunkedStep step = s.schedule_chunked(budget);
            ASSERT_FALSE(step.empty());
            ASSERT_LE(step.num_tokens(), budget);
            batch = build_chunked_batch(step, s, kMaxBlocks);
            sampled = step.sampled;
            drafts = step.drafts;
        } else {
            const Step step = s.schedule();
            ASSERT_FALSE(step.seqs.empty());
            batch = build_paged_batch(step, s, kMaxBlocks);
            sampled = step.seqs;
            drafts = step.drafts;
        }
        ASSERT_LE(batch.num_seqs, c.max_batch);
        const std::vector<SampleRow> rows = sample_rows(sampled, s, drafts);
        ASSERT_EQ(int64_t(rows.size()), batch.num_seqs);
        s.update(model.run(batch, rows));
        ++out.steps;
        for (const Request& r : s.take_finished()) {
            out.done[r.id] = r.output;
            out.preemptions += r.preemptions;
        }
    }
    out.drafts = s.draft_counts();
    EXPECT_EQ(s.blocks().num_free(), c.num_blocks - 1);
}

}  // namespace

TEST(SchedulerSpeculative, OffByDefault) {
    Scheduler s(config(40, 4, 0));
    s.submit(make(1, repeating(30), 20));
    s.schedule();
    s.update(Tokens{11});
    const Step step = s.schedule();
    EXPECT_FALSE(step.prefill);
    EXPECT_TRUE(step.drafts.empty());
    s.update(Tokens{12});
    EXPECT_EQ(s.draft_counts().proposed, 0);
}

TEST(SchedulerSpeculative, RejectsBadConfig) {
    SchedulerConfig c = config(40, 4, -1);
    EXPECT_THROW(Scheduler{c}, std::invalid_argument);
    c = config(40, 4, 4);
    c.max_ngram = 0;
    EXPECT_THROW(Scheduler{c}, std::invalid_argument);
}

TEST(SchedulerSpeculative, ADecodeStepCarriesTheLookupDraft) {
    Scheduler s(config(40, 8, 4));
    const Request r = make(1, repeating(30), 20);
    s.submit(r);
    EXPECT_TRUE(s.schedule().prefill);
    s.update(Tokens{11});

    const Step step = s.schedule();
    ASSERT_EQ(step.seqs, (std::vector<SeqId>{1}));
    ASSERT_EQ(step.drafts.size(), 1u);
    Tokens context = r.prompt;
    context.push_back(11);
    EXPECT_EQ(step.drafts[0], engine::propose_prompt_lookup(context, 3, 4));
    EXPECT_EQ(step.drafts[0], (Tokens{12, 13, 14, 15}));
    EXPECT_EQ(s.blocks().length(1), 31 + 4);

    const PagedBatch batch = build_paged_batch(step, s, kMaxBlocks);
    EXPECT_EQ(batch.num_seqs, 5);
    EXPECT_EQ(batch.token_ids, (Tokens{11, 12, 13, 14, 15}));
    EXPECT_EQ(batch.positions, (Tokens{30, 31, 32, 33, 34}));
    EXPECT_EQ(batch.context_lens, (Tokens{31, 32, 33, 34, 35}));
    EXPECT_EQ(batch.logit_rows, (Tokens{0, 1, 2, 3, 4}));
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(batch.slot_mapping[size_t(i)], s.blocks().slot(1, 30 + i));

    const std::vector<SampleRow> rows = sample_rows(step.seqs, s, step.drafts);
    ASSERT_EQ(rows.size(), 5u);
    for (size_t i = 0; i < rows.size(); ++i) EXPECT_EQ(rows[i].step, 1 + i);
}

TEST(SchedulerSpeculative, UpdateKeepsTheAgreedDraftsAndGivesTheRestBack) {
    for (int agreed = 0; agreed <= 4; ++agreed) {
        Scheduler s(config(40, 8, 4));
        s.submit(make(1, repeating(30), 20));
        s.schedule();
        s.update(Tokens{11});
        const Step step = s.schedule();
        ASSERT_EQ(step.drafts[0], (Tokens{12, 13, 14, 15}));

        Tokens sampled{12, 13, 14, 15, 16};
        if (agreed < 4) sampled[size_t(agreed)] = 99;
        s.update(sampled);

        const Request& r = s.request(1);
        ASSERT_EQ(int(r.output.size()), 1 + agreed + 1) << "agreed " << agreed;
        EXPECT_EQ(r.output.back(), agreed < 4 ? 99 : 16);
        EXPECT_EQ(s.blocks().length(1), r.length() - 1);
        EXPECT_EQ(int(s.blocks().table(1).size()), (int(r.length()) - 1 + kBlock - 1) / kBlock);
        EXPECT_EQ(s.draft_counts().proposed, 4);
        EXPECT_EQ(s.draft_counts().accepted, agreed);

        const Step next = s.schedule();
        const PagedBatch batch = build_paged_batch(next, s, kMaxBlocks);
        EXPECT_EQ(batch.token_ids.front(), r.output.back());
        EXPECT_EQ(batch.positions.front(), int32_t(r.length() - 1));
    }
}

TEST(SchedulerSpeculative, UpdateNeedsOneTokenPerSampledRow) {
    Scheduler s(config(40, 8, 4));
    s.submit(make(1, repeating(30), 20));
    s.schedule();
    s.update(Tokens{11});
    s.schedule();
    EXPECT_THROW(s.update(Tokens{12}), std::invalid_argument);
    EXPECT_THROW(s.update(Tokens{12, 13, 14, 15, 16, 17}), std::invalid_argument);
    s.update(Tokens{12, 13, 14, 15, 16});
    EXPECT_EQ(s.request(1).output.size(), 6u);
}

TEST(SchedulerSpeculative, EosInsideTheAgreedRunEndsTheRequest) {
    Scheduler s(config(40, 8, 4, kEos));
    Tokens prompt = repeating(20);
    prompt[13] = kEos;
    prompt.insert(prompt.end(), {11, 12});
    s.submit(make(1, prompt, 20));
    s.schedule();
    s.update(Tokens{13});
    const Step step = s.schedule();
    ASSERT_EQ(step.drafts[0], (Tokens{kEos, 15, 16, 17}));
    s.update(Tokens{kEos, 15, 16, 17, 18});
    const std::vector<Request> finished = s.take_finished();
    ASSERT_EQ(finished.size(), 1u);
    EXPECT_EQ(finished[0].output, (Tokens{13, kEos}));
    EXPECT_EQ(s.blocks().num_free(), 39);
    EXPECT_EQ(s.draft_counts().accepted, 0);
}

TEST(SchedulerSpeculative, ADraftNeverPassesMaxNewTokens) {
    Scheduler s(config(40, 8, 8));
    s.submit(make(1, repeating(30), 4));
    s.schedule();
    s.update(Tokens{11});
    const Step step = s.schedule();
    ASSERT_EQ(step.drafts[0], (Tokens{12, 13}));
    s.update(Tokens{12, 13, 14});
    const std::vector<Request> finished = s.take_finished();
    ASSERT_EQ(finished.size(), 1u);
    EXPECT_EQ(finished[0].output, (Tokens{11, 12, 13, 14}));

    Scheduler last(config(40, 8, 8));
    last.submit(make(2, repeating(30), 2));
    last.schedule();
    last.update(Tokens{11});
    EXPECT_TRUE(last.schedule().drafts[0].empty());
}

TEST(SchedulerSpeculative, DraftsOnlyTakeRowsNoSequenceNeeds) {
    Scheduler s(config(80, 4, 8));
    for (SeqId id = 1; id <= 3; ++id) s.submit(make(id, repeating(30), 20));
    s.schedule();
    s.update(Tokens{11, 11, 11});
    const Step step = s.schedule();
    ASSERT_EQ(step.seqs.size(), 3u);
    ASSERT_EQ(step.drafts.size(), 3u);
    EXPECT_EQ(step.drafts[0].size(), 1u);
    EXPECT_TRUE(step.drafts[1].empty());
    EXPECT_TRUE(step.drafts[2].empty());
    EXPECT_EQ(build_paged_batch(step, s, kMaxBlocks).num_seqs, 4);
    s.update(Tokens{12, 13, 12, 12});

    s.submit(make(4, repeating(30), 20));
    s.schedule();
    s.update(Tokens{11});
    const Step full = s.schedule();
    ASSERT_EQ(full.seqs.size(), 4u);
    for (const Tokens& draft : full.drafts) EXPECT_TRUE(draft.empty());
}

TEST(SchedulerSpeculative, AChunkedStepSpendsItsBudgetOnPrefillFirst) {
    Scheduler s(config(80, 8, 8));
    s.submit(make(1, repeating(30), 20));
    ChunkedStep step = s.schedule_chunked(32);
    s.update(Tokens{11});

    s.submit(make(2, repeating(60), 20));
    step = s.schedule_chunked(32);
    ASSERT_EQ(step.decode, (std::vector<SeqId>{1}));
    ASSERT_EQ(step.prefill.size(), 1u);
    EXPECT_EQ(step.prefill[0].count, 31);
    ASSERT_EQ(step.drafts.size(), 1u);
    EXPECT_TRUE(step.drafts[0].empty());
    EXPECT_EQ(step.num_tokens(), 32);
    s.update(Tokens{12});

    step = s.schedule_chunked(32);
    ASSERT_EQ(step.decode, (std::vector<SeqId>{1}));
    ASSERT_EQ(step.prefill.size(), 1u);
    EXPECT_EQ(step.prefill[0].count, 29);
    EXPECT_EQ(step.drafts[0], (Tokens{13, 14}));
    EXPECT_EQ(step.sampled, (std::vector<SeqId>{1, 2}));
    const PagedBatch batch = build_chunked_batch(step, s, kMaxBlocks);
    EXPECT_EQ(batch.num_tokens(), 32);
    EXPECT_EQ(batch.decode_rows(), 3);
    EXPECT_EQ(batch.logit_rows, (Tokens{0, 1, 2, 31}));
    EXPECT_EQ(batch.num_seqs, 4);
    EXPECT_EQ(sample_rows(step.sampled, s, step.drafts).size(), 4u);
    s.update(Tokens{13, 14, 15, 11});
    EXPECT_EQ(s.request(1).output, (Tokens{11, 12, 13, 14, 15}));
    EXPECT_EQ(s.request(2).output, (Tokens{11}));
}

TEST(SchedulerSpeculative, AShortPoolShortensTheDraftAndPreemptsNobody) {
    SchedulerConfig c = config(5, 8, 8);
    c.max_context = c.max_prefill_tokens = 64;
    Scheduler s(c);
    s.submit(make(1, repeating(30), 20));
    s.submit(make(2, repeating(14), 40));
    s.submit(make(3, repeating(5), 20));
    s.schedule();
    s.update(Tokens{11, 15, 16});
    ASSERT_EQ(s.blocks().num_free(), 0);

    const Step step = s.schedule();
    ASSERT_EQ(step.seqs, (std::vector<SeqId>{1, 2, 3}));
    EXPECT_EQ(step.drafts[0], (Tokens{12}));
    EXPECT_EQ(step.drafts[1], (Tokens{16}));
    EXPECT_TRUE(step.drafts[2].empty());
    EXPECT_EQ(s.blocks().length(1), 32);
    EXPECT_EQ(s.blocks().length(2), 16);
    EXPECT_EQ(s.blocks().num_free(), 0);
    for (const SeqId id : step.seqs) EXPECT_EQ(s.request(id).preemptions, 0);
    s.update(Tokens{12, 13, 16, 17, 7});
    EXPECT_EQ(s.request(1).output, (Tokens{11, 12, 13}));
    EXPECT_EQ(s.request(2).output, (Tokens{15, 16, 17}));
}

TEST(SchedulerSpeculative, AnswersMatchPlainDecodingForEveryDraftLength) {
    std::mt19937 rng(7);
    std::vector<Request> requests;
    for (int i = 0; i < 12; ++i) {
        const int prompt = std::uniform_int_distribution<int>(1, 120)(rng);
        const int fresh = std::uniform_int_distribution<int>(1, kContext - prompt)(rng);
        requests.push_back(make(i, random_prompt(rng, prompt), fresh, uint64_t(i % 4)));
    }
    int64_t accepted = 0;
    for (const int max_draft : {0, 1, 3, 8, 16}) {
        for (const int max_batch : {1, 3, 8, 16}) {
            for (const int budget : {0, max_batch, max_batch + 24}) {
                Served out;
                serve(config(400, max_batch, max_draft, kEos), requests, budget, out);
                if (HasFatalFailure()) return;
                ASSERT_EQ(out.done.size(), requests.size());
                for (const Request& r : requests)
                    EXPECT_EQ(out.done.at(r.id), solo(r, kEos))
                        << "draft " << max_draft << " batch " << max_batch << " budget " << budget;
                EXPECT_LE(out.drafts.accepted, out.drafts.proposed);
                if (max_draft == 0) {
                    EXPECT_EQ(out.drafts.proposed, 0);
                }
                accepted += out.drafts.accepted;
            }
        }
    }
    EXPECT_GT(accepted, 0);
}

TEST(SchedulerSpeculative, AcceptedDraftsSaveSteps) {
    std::vector<Request> requests{make(1, repeating(60), 100)};
    Served plain, speculative;
    serve(config(400, 8, 0), requests, 0, plain);
    serve(config(400, 8, 7), requests, 0, speculative);
    if (HasFatalFailure()) return;
    EXPECT_EQ(plain.done, speculative.done);
    EXPECT_EQ(plain.steps, 100);
    EXPECT_LT(speculative.steps, plain.steps);
    EXPECT_GT(speculative.drafts.accepted, 0);
}

TEST(SchedulerSpeculative, AdaptiveLimitHalvesOnARejectionAndDoublesBack) {
    SchedulerConfig c = config(40, 16, 8);
    c.adaptive_draft = true;
    Scheduler s(c);
    s.submit(make(1, repeating(30), 100));
    s.schedule();
    s.update(Tokens{11});

    Tokens draft = s.schedule().drafts.at(0);
    ASSERT_EQ(draft.size(), 8u);
    Tokens reply = draft;
    reply.back() = 11;
    reply.push_back(0);
    s.update(reply);

    draft = s.schedule().drafts.at(0);
    ASSERT_EQ(draft.size(), 4u);
    reply = draft;
    reply.push_back(16);
    s.update(reply);

    EXPECT_EQ(s.schedule().drafts.at(0).size(), 8u);
}

TEST(SchedulerSpeculative, FixedLimitStaysAfterARejection) {
    Scheduler s(config(40, 16, 8));
    s.submit(make(1, repeating(30), 100));
    s.schedule();
    s.update(Tokens{11});
    Tokens reply = s.schedule().drafts.at(0);
    ASSERT_EQ(reply.size(), 8u);
    reply.back() = 11;
    reply.push_back(0);
    s.update(reply);
    EXPECT_EQ(s.schedule().drafts.at(0).size(), 8u);
}

TEST(SchedulerSpeculative, ADraftSourceReplacesTheLookupAndIsCutToTheRowsLeft) {
    Scheduler s(config(40, 16, 4));
    int asked = 0;
    s.set_draft_source([&asked](SeqId id, std::span<const int32_t> context, int want) {
        EXPECT_EQ(id, 1);
        EXPECT_EQ(context.size(), 31u);
        asked = want;
        return Tokens(9, 7);
    });
    s.submit(make(1, repeating(30), 100));
    s.schedule();
    s.update(Tokens{11});
    EXPECT_EQ(s.schedule().drafts.at(0), Tokens(4, 7));
    EXPECT_EQ(asked, 4);
}

TEST(SchedulerSpeculative, AnswersMatchPlainDecodingWithAnyDraftSource) {
    std::mt19937 rng(11);
    std::vector<Request> requests;
    for (int i = 0; i < 12; ++i) {
        const int prompt = std::uniform_int_distribution<int>(1, 120)(rng);
        const int fresh = std::uniform_int_distribution<int>(1, kContext - prompt)(rng);
        requests.push_back(make(i, random_prompt(rng, prompt), fresh, uint64_t(i % 4)));
    }
    int64_t accepted = 0;
    for (const int budget : {0, 8, 32}) {
        Served out;
        serve(config(400, 8, 5, kEos), requests, budget, out,
              [&rng](SeqId, std::span<const int32_t>, int want) {
                  Tokens guess;
                  for (int i = 0; i < want; ++i)
                      guess.push_back(3 + int32_t(std::uniform_int_distribution<int>(0, 11)(rng)));
                  return guess;
              });
        if (HasFatalFailure()) return;
        ASSERT_EQ(out.done.size(), requests.size());
        for (const Request& r : requests) EXPECT_EQ(out.done.at(r.id), solo(r, kEos));
        EXPECT_GT(out.drafts.proposed, 0);
        accepted += out.drafts.accepted;
    }
    EXPECT_GT(accepted, 0);
}

TEST(SchedulerSpeculative, RandomRunsWithPreemptionMatchPlainDecoding) {
    int preempted = 0;
    int64_t accepted = 0;
    for (int seed = 0; seed < 40; ++seed) {
        std::mt19937 rng(uint32_t(seed) + 23);
        const auto pick = [&rng](int lo, int hi) {
            return std::uniform_int_distribution<int>(lo, hi)(rng);
        };
        SchedulerConfig c = config(pick(18, 50), pick(1, 8), pick(1, 12), kEos);
        c.max_ngram = pick(1, 4);
        c.adaptive_draft = seed % 2 == 1;
        const int budget = pick(0, 1) == 0 ? 0 : c.max_batch + pick(0, 64);
        std::vector<Request> requests;
        for (int i = 0; i < 16; ++i) {
            const int prompt = pick(1, 180);
            requests.push_back(make(i, random_prompt(rng, prompt), pick(1, kContext - prompt),
                                    uint64_t(pick(0, 4))));
        }
        Served out;
        serve(c, requests, budget, out);
        if (HasFatalFailure()) return;
        ASSERT_EQ(out.done.size(), requests.size()) << "seed " << seed;
        for (const Request& r : requests)
            EXPECT_EQ(out.done.at(r.id), solo(r, kEos)) << "seed " << seed;
        preempted += out.preemptions;
        accepted += out.drafts.accepted;
    }
    EXPECT_GT(preempted, 0);
    EXPECT_GT(accepted, 0);
}
