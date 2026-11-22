#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "engine/sched/paged_batch.h"
#include "engine/sched/scheduler.h"

using engine::ChunkedStep;
using engine::PagedBatch;
using engine::PrefillChunk;
using engine::PrefixCounts;
using engine::Request;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;
using engine::Step;

namespace {

constexpr int kBlock = 16;
constexpr int kContext = 256;
constexpr int kMaxBlocks = kContext / kBlock;

using Tokens = std::vector<int32_t>;

SchedulerConfig config(bool prefix_cache, int num_blocks = 64, int max_draft = 0,
                       int context = kContext) {
    return {.num_blocks = num_blocks,
            .block_size = kBlock,
            .max_batch = 8,
            .max_prefill_tokens = context,
            .max_context = context,
            .max_draft = max_draft,
            .prefix_cache = prefix_cache};
}

Tokens run(int count, int32_t first) {
    Tokens tokens(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) tokens[size_t(i)] = first + i;
    return tokens;
}

Tokens join(Tokens a, const Tokens& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

Tokens head(const Tokens& tokens, int count) {
    return Tokens(tokens.begin(), tokens.begin() + count);
}

Tokens tokens_of(const Request& r) {
    return join(r.prompt, r.output);
}

Request make(SeqId id, Tokens prompt, int max_new_tokens) {
    return Request{.id = id, .prompt = std::move(prompt), .max_new_tokens = max_new_tokens};
}

std::vector<int> first_blocks(const Scheduler& s, SeqId id, int count) {
    const std::vector<int>& table = s.blocks().table(id);
    return std::vector<int>(table.begin(), table.begin() + count);
}

void prefill(Scheduler& s, SeqId id, Tokens prompt, int max_new_tokens) {
    s.submit(make(id, std::move(prompt), max_new_tokens));
    const Step step = s.schedule();
    ASSERT_TRUE(step.prefill);
    ASSERT_EQ(step.seqs, std::vector<SeqId>{id});
    s.update(Tokens{7});
}

TEST(PrefixCacheScheduler, ALaterPromptSkipsTheBlocksItShares) {
    Scheduler s(config(true));
    const Tokens system = run(32, 100);
    prefill(s, 1, join(system, run(8, 300)), 4);
    s.submit(make(2, join(system, run(13, 500)), 4));
    const Step step = s.schedule();
    ASSERT_TRUE(step.prefill);
    ASSERT_EQ(step.seqs, std::vector<SeqId>{2});
    EXPECT_EQ(step.cached, std::vector<int>{32});
    EXPECT_EQ(first_blocks(s, 2, 2), first_blocks(s, 1, 2));
    EXPECT_EQ(s.blocks().ref_count(s.blocks().table(2)[0]), 2);
    const PagedBatch batch = build_paged_batch(step, s, kMaxBlocks);
    EXPECT_EQ(batch.num_tokens(), 13);
    EXPECT_EQ(batch.positions.front(), 32);
    EXPECT_EQ(batch.token_ids.front(), 500);
    EXPECT_EQ(batch.cached_lens, std::vector<int32_t>{32});
    EXPECT_EQ(batch.logit_rows, std::vector<int32_t>{12});
    EXPECT_EQ(s.prefix_counts().prefill, 40 + 45);
    EXPECT_EQ(s.prefix_counts().cached, 32);
}

TEST(PrefixCacheScheduler, TheCacheIsOffByDefault) {
    Scheduler s(SchedulerConfig{.num_blocks = 64,
                                .block_size = kBlock,
                                .max_batch = 8,
                                .max_prefill_tokens = kContext,
                                .max_context = kContext});
    const Tokens system = run(32, 100);
    prefill(s, 1, join(system, run(8, 300)), 4);
    s.submit(make(2, join(system, run(13, 500)), 4));
    const Step step = s.schedule();
    EXPECT_EQ(step.cached, std::vector<int>{0});
    EXPECT_NE(s.blocks().table(2)[0], s.blocks().table(1)[0]);
    EXPECT_EQ(build_paged_batch(step, s, kMaxBlocks).num_tokens(), 45);
    EXPECT_EQ(s.prefix_counts().prefill, 40 + 45);
    EXPECT_EQ(s.prefix_counts().cached, 0);
}

TEST(PrefixCacheScheduler, RequestsAdmittedInTheSameStepDoNotShare) {
    Scheduler s(config(true));
    const Tokens prompt = run(40, 100);
    s.submit(make(1, prompt, 4));
    s.submit(make(2, prompt, 4));
    const Step step = s.schedule();
    EXPECT_EQ(step.cached, (std::vector<int>{0, 0}));
    EXPECT_NE(s.blocks().table(1)[0], s.blocks().table(2)[0]);
    EXPECT_EQ(s.blocks().match_prefix(prompt), 0);
    s.update(Tokens{7, 7});
    EXPECT_EQ(s.blocks().match_prefix(prompt), 32);
    s.submit(make(3, prompt, 4));
    EXPECT_EQ(s.schedule().cached, std::vector<int>{32});
    EXPECT_EQ(first_blocks(s, 3, 2), first_blocks(s, 1, 2));
}

TEST(PrefixCacheScheduler, ARepeatedPromptReusesAllButTheBlockOfItsLastToken) {
    Scheduler s(config(true));
    const Tokens prompt = run(32, 100);
    prefill(s, 1, prompt, 1);
    ASSERT_EQ(s.take_finished().size(), 1u);
    s.submit(make(2, prompt, 1));
    const Step step = s.schedule();
    EXPECT_EQ(step.cached, std::vector<int>{16});
    EXPECT_EQ(build_paged_batch(step, s, kMaxBlocks).num_tokens(), 16);
}

TEST(PrefixCacheScheduler, ChunkedPrefillStartsAtTheCachedCount) {
    Scheduler s(config(true));
    const Tokens system = run(32, 100);
    s.submit(make(1, join(system, run(8, 300)), 6));
    ChunkedStep step = s.schedule_chunked(64);
    ASSERT_EQ(step.prefill.size(), 1u);
    EXPECT_EQ(step.prefill[0].first, 0);
    s.update(Tokens{7});
    s.submit(make(2, join(system, run(13, 500)), 4));
    step = s.schedule_chunked(8);
    EXPECT_EQ(step.decode, std::vector<SeqId>{1});
    ASSERT_EQ(step.prefill.size(), 1u);
    EXPECT_EQ(step.prefill[0].id, 2);
    EXPECT_EQ(step.prefill[0].first, 32);
    EXPECT_EQ(step.prefill[0].count, 7);
    EXPECT_EQ(step.sampled, std::vector<SeqId>{1});
    PagedBatch batch = build_chunked_batch(step, s, kMaxBlocks);
    EXPECT_EQ(batch.cached_lens, std::vector<int32_t>{32});
    EXPECT_EQ(batch.num_tokens(), 8);
    s.update(Tokens{7});
    step = s.schedule_chunked(8);
    ASSERT_EQ(step.prefill.size(), 1u);
    EXPECT_EQ(step.prefill[0].first, 39);
    EXPECT_EQ(step.prefill[0].count, 6);
    EXPECT_EQ(step.sampled, (std::vector<SeqId>{1, 2}));
}

TEST(PrefixCacheScheduler, OnlyRowsThatAreNotCachedCountTowardsMaxPrefillTokens) {
    for (const bool cache : {false, true}) {
        SCOPED_TRACE(cache ? "cache on" : "cache off");
        Scheduler s(config(cache, 64, 0, 64));
        const Tokens first = run(60, 100);
        prefill(s, 1, first, 1);
        ASSERT_EQ(s.take_finished().size(), 1u);
        s.submit(make(2, join(head(first, 48), run(12, 500)), 1));
        s.submit(make(3, join(head(first, 48), run(12, 700)), 1));
        const Step step = s.schedule();
        EXPECT_EQ(step.seqs, cache ? (std::vector<SeqId>{2, 3}) : (std::vector<SeqId>{2}));
        EXPECT_EQ(build_paged_batch(step, s, 4).num_tokens(), cache ? 24 : 60);
    }
}

TEST(PrefixCacheScheduler, GeneratedTokensArePublishedOnceTheirBlockIsWritten) {
    Scheduler s(config(true));
    prefill(s, 1, run(10, 100), 40);
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ(s.schedule().seqs, std::vector<SeqId>{1});
        s.update(Tokens{int32_t(200 + i)});
        const Tokens next = join(tokens_of(s.request(1)), run(3, 900));
        EXPECT_EQ(s.blocks().match_prefix(next), i < 5 ? 0 : 16);
    }
}

TEST(PrefixCacheScheduler, AFinalStepWithARejectedDraftPublishesOnlyWrittenTokens) {
    SchedulerConfig c = config(true, 64, 3);
    c.eos_id = 99;
    Scheduler s(c);
    const Tokens prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1};
    prefill(s, 1, prompt, 10);
    const Step step = s.schedule();
    ASSERT_EQ(step.drafts, (std::vector<Tokens>{{8, 9, 10}}));
    s.update(Tokens{99, 8, 9, 10});
    ASSERT_EQ(s.take_finished().size(), 1u);
    EXPECT_EQ(s.blocks().match_prefix(join(prompt, Tokens{7, 99, 5})), 0);
}

uint64_t mix(uint64_t h, int32_t token) {
    uint64_t x = h ^ (uint64_t(uint32_t(token)) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::vector<Tokens> rows_of(const Scheduler& s, const Step& step) {
    std::vector<Tokens> rows;
    for (size_t i = 0; i < step.seqs.size(); ++i) {
        const Tokens draft = i < step.drafts.size() ? step.drafts[i] : Tokens{};
        const Tokens tokens = join(tokens_of(s.request(step.seqs[i])), draft);
        const int count =
            step.prefill ? s.blocks().length(step.seqs[i]) - step.cached[i] : 1 + int(draft.size());
        rows.insert(rows.end(), size_t(count), tokens);
    }
    return rows;
}

std::vector<Tokens> rows_of(const Scheduler& s, const ChunkedStep& step) {
    std::vector<Tokens> rows;
    for (size_t i = 0; i < step.decode.size(); ++i) {
        const Tokens draft = i < step.drafts.size() ? step.drafts[i] : Tokens{};
        rows.insert(rows.end(), 1 + draft.size(),
                    join(tokens_of(s.request(step.decode[i])), draft));
    }
    for (const PrefillChunk& chunk : step.prefill)
        rows.insert(rows.end(), size_t(chunk.count), tokens_of(s.request(chunk.id)));
    return rows;
}

class FakeModel {
public:
    explicit FakeModel(int num_blocks) : kv_(size_t(num_blocks) * kBlock, 0) {}

    Tokens run(const PagedBatch& batch, const std::vector<Tokens>& rows, const Scheduler& s) {
        if (int64_t(rows.size()) != batch.num_tokens()) {
            ADD_FAILURE() << "the step has " << rows.size() << " rows, the batch "
                          << batch.num_tokens();
            return {};
        }
        for (int64_t r = 0; r < batch.num_tokens(); ++r) {
            const int pos = batch.positions[size_t(r)];
            const int block = batch.block_tables[size_t(r * batch.max_blocks + pos / kBlock)];
            if (batch.slot_mapping[size_t(r)] != block * kBlock + pos % kBlock ||
                batch.token_ids[size_t(r)] != rows[size_t(r)][size_t(pos)]) {
                ADD_FAILURE() << "row " << r << " has the wrong slot or token";
                return {};
            }
            EXPECT_EQ(s.blocks().ref_count(block), 1) << "row " << r << " writes a shared block";
            kv_[size_t(block * kBlock + pos % kBlock)] =
                mix(pos == 0 ? 0 : kv_[slot(batch, r, pos - 1)], batch.token_ids[size_t(r)]);
        }
        for (int64_t r = 0; r < batch.num_tokens(); ++r) {
            uint64_t want = 0;
            for (int p = 0; p <= batch.positions[size_t(r)]; ++p) {
                want = mix(want, rows[size_t(r)][size_t(p)]);
                if (kv_[slot(batch, r, p)] != want) {
                    ADD_FAILURE() << "row " << r << " reads a key written for other tokens at "
                                  << p;
                    return {};
                }
            }
        }
        Tokens sampled;
        for (const int32_t r : batch.logit_rows)
            sampled.push_back(3 + int32_t(kv_[size_t(batch.slot_mapping[size_t(r)])] % 5));
        return sampled;
    }

private:
    size_t slot(const PagedBatch& batch, int64_t row, int pos) const {
        const int block = batch.block_tables[size_t(row * batch.max_blocks + pos / kBlock)];
        return size_t(block * kBlock + pos % kBlock);
    }

    std::vector<uint64_t> kv_;
};

struct Outcome {
    std::map<SeqId, Tokens> outputs;
    PrefixCounts counts;
    int preemptions = 0;
};

Outcome serve(const SchedulerConfig& c, const std::vector<Request>& requests, int budget) {
    Scheduler s(c);
    FakeModel model(c.num_blocks);
    Outcome out;
    size_t submitted = 0;
    for (int step = 0; out.outputs.size() < requests.size(); ++step) {
        if (step > 20000) {
            ADD_FAILURE() << "the requests did not finish";
            break;
        }
        if (submitted < requests.size() && step % 2 == 0) s.submit(requests[submitted++]);
        PagedBatch batch;
        std::vector<Tokens> rows;
        if (budget > 0) {
            const ChunkedStep chunked = s.schedule_chunked(budget);
            if (chunked.empty()) continue;
            batch = build_chunked_batch(chunked, s, kMaxBlocks);
            rows = rows_of(s, chunked);
        } else {
            const Step whole = s.schedule();
            if (whole.seqs.empty()) continue;
            batch = build_paged_batch(whole, s, kMaxBlocks);
            rows = rows_of(s, whole);
        }
        const Tokens sampled = model.run(batch, rows, s);
        if (::testing::Test::HasFailure()) break;
        s.update(sampled);
        for (const Request& r : s.take_finished()) {
            out.outputs[r.id] = r.output;
            out.preemptions += r.preemptions;
        }
    }
    out.counts = s.prefix_counts();
    return out;
}

Tokens predict(const Tokens& prompt, int max_new_tokens) {
    uint64_t h = 0;
    for (const int32_t token : prompt) h = mix(h, token);
    Tokens out;
    while (int(out.size()) < max_new_tokens) {
        out.push_back(3 + int32_t(h % 5));
        h = mix(h, out.back());
    }
    return out;
}

std::vector<Request> traffic(uint32_t seed) {
    std::mt19937 rng(seed);
    const auto pick = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    const auto random_tokens = [&](int count) {
        Tokens tokens;
        for (int i = 0; i < count; ++i) tokens.push_back(3 + pick(0, 4));
        return tokens;
    };
    const std::vector<Tokens> systems{{}, random_tokens(20), random_tokens(37), random_tokens(64)};
    std::vector<Request> requests;
    for (int i = 0; i < 40; ++i) {
        const int kind = i > 0 ? pick(0, 3) : 0;
        const Request& earlier = requests.empty() ? Request{} : requests[size_t(pick(0, i - 1))];
        Tokens prompt = join(systems[size_t(pick(0, 3))], random_tokens(pick(1, 40)));
        if (kind == 1) prompt = earlier.prompt;
        if (kind == 2) {
            const Tokens turn =
                join(join(earlier.prompt, predict(earlier.prompt, earlier.max_new_tokens)),
                     random_tokens(pick(1, 10)));
            if (turn.size() + 30 <= size_t(kContext)) prompt = turn;
        }
        requests.push_back(make(i, std::move(prompt), pick(1, 30)));
    }
    return requests;
}

TEST(PrefixCacheScheduler, RandomTrafficReadsOnlyKeysWrittenForItsOwnTokens) {
    for (const int num_blocks : {64, 18}) {
        for (const int budget : {0, 16, 40}) {
            for (const int max_draft : {0, 3}) {
                SCOPED_TRACE("blocks " + std::to_string(num_blocks) + ", budget " +
                             std::to_string(budget) + ", drafts " + std::to_string(max_draft));
                const std::vector<Request> requests =
                    traffic(uint32_t(num_blocks * 100 + budget * 10 + max_draft));
                const Outcome off = serve(config(false, num_blocks, max_draft), requests, budget);
                const Outcome on = serve(config(true, num_blocks, max_draft), requests, budget);
                if (HasFailure()) return;
                EXPECT_EQ(on.outputs, off.outputs);
                for (const Request& r : requests)
                    EXPECT_EQ(on.outputs.at(r.id), predict(r.prompt, r.max_new_tokens)) << r.id;
                EXPECT_EQ(off.counts.cached, 0);
                EXPECT_GT(on.counts.cached, on.counts.prefill / 10);
                if (num_blocks == 18) {
                    EXPECT_GT(on.preemptions, 0);
                }
            }
        }
    }
}

}  // namespace
