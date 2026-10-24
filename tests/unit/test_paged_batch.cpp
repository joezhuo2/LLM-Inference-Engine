#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine/sched/paged_batch.h"

using engine::attention_splits;
using engine::build_paged_batch;
using engine::pad_decode_batch;
using engine::PagedBatch;
using engine::Request;
using engine::sample_rows;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;
using engine::Step;

namespace {

constexpr int kBlockSize = 4;
constexpr int kMaxBlocks = 8;

SchedulerConfig config(int num_blocks = 40) {
    return SchedulerConfig{.num_blocks = num_blocks,
                           .block_size = kBlockSize,
                           .max_batch = 4,
                           .max_prefill_tokens = 32,
                           .max_context = 32};
}

Request make(SeqId id, int prompt_len, int max_new_tokens, uint64_t seed = 0) {
    Request r{.id = id, .max_new_tokens = max_new_tokens, .params = {0.5f, seed}};
    for (int i = 0; i < prompt_len; ++i) r.prompt.push_back(int32_t(100 * id + i));
    return r;
}

}  // namespace

TEST(PagedBatch, PrefillHasOneRowPerPromptTokenWithCausalContextLengths) {
    Scheduler s(config());
    s.submit(make(1, 6, 4));
    s.submit(make(2, 3, 4));
    const Step step = s.schedule();
    ASSERT_TRUE(step.prefill);
    const PagedBatch b = build_paged_batch(step, s, kMaxBlocks);

    EXPECT_TRUE(b.prefill);
    EXPECT_EQ(b.num_seqs, 2);
    EXPECT_EQ(b.max_blocks, kMaxBlocks);
    ASSERT_EQ(b.num_tokens(), 9);
    EXPECT_EQ(b.token_ids, (std::vector<int32_t>{100, 101, 102, 103, 104, 105, 200, 201, 202}));
    EXPECT_EQ(b.positions, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 0, 1, 2}));
    EXPECT_EQ(b.context_lens, (std::vector<int32_t>{1, 2, 3, 4, 5, 6, 1, 2, 3}));
    EXPECT_EQ(b.logit_rows, (std::vector<int32_t>{5, 8}));
    for (int64_t t = 0; t < b.num_tokens(); ++t) {
        const SeqId id = t < 6 ? 1 : 2;
        EXPECT_EQ(b.slot_mapping[size_t(t)], s.blocks().slot(id, b.positions[size_t(t)]));
    }
}

TEST(PagedBatch, EveryRowCarriesItsSequencesBlockTablePaddedWithTheNullBlock) {
    Scheduler s(config());
    s.submit(make(1, 6, 4));
    const Step step = s.schedule();
    const PagedBatch b = build_paged_batch(step, s, kMaxBlocks);
    const std::vector<int>& table = s.blocks().table(1);
    ASSERT_EQ(table.size(), 2u);
    ASSERT_EQ(b.block_tables.size(), size_t(6 * kMaxBlocks));
    for (int row = 0; row < 6; ++row) {
        const int32_t* r = b.block_tables.data() + row * kMaxBlocks;
        EXPECT_EQ(r[0], table[0]);
        EXPECT_EQ(r[1], table[1]);
        for (int c = 2; c < kMaxBlocks; ++c) EXPECT_EQ(r[c], 0);
    }
}

TEST(PagedBatch, DecodeBatchHasOneRowPerSequence) {
    Scheduler s(config());
    s.submit(make(1, 4, 5));
    s.submit(make(2, 5, 5));
    ASSERT_TRUE(s.schedule().prefill);
    s.update(std::vector<int32_t>{901, 902});
    const Step step = s.schedule();
    ASSERT_FALSE(step.prefill);
    const PagedBatch b = build_paged_batch(step, s, kMaxBlocks);

    EXPECT_FALSE(b.prefill);
    EXPECT_EQ(b.num_seqs, 2);
    EXPECT_EQ(b.token_ids, (std::vector<int32_t>{901, 902}));
    EXPECT_EQ(b.positions, (std::vector<int32_t>{4, 5}));
    EXPECT_EQ(b.context_lens, (std::vector<int32_t>{5, 6}));
    EXPECT_EQ(b.logit_rows, (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(b.slot_mapping[0], s.blocks().slot(1, 4));
    EXPECT_EQ(b.slot_mapping[1], s.blocks().slot(2, 5));
    EXPECT_EQ(b.block_tables.size(), size_t(2 * kMaxBlocks));
}

TEST(PagedBatch, ARecomputedSequencePrefillsItsPromptAndItsOutput) {
    SchedulerConfig c = config(7);
    c.max_batch = 2;
    c.max_context = 16;
    c.max_prefill_tokens = 16;
    Scheduler s(c);
    s.submit(make(1, 7, 8));
    s.submit(make(2, 7, 8));
    Step step;
    bool recomputed = false;
    for (int i = 0; i < 40 && !recomputed; ++i) {
        step = s.schedule();
        ASSERT_FALSE(step.seqs.empty());
        recomputed = step.prefill && s.request(step.seqs[0]).preemptions > 0;
        if (!recomputed) s.update(std::vector<int32_t>(step.seqs.size(), 600 + i));
    }
    ASSERT_TRUE(recomputed);
    ASSERT_EQ(step.seqs.size(), 1u);
    const Request& r = s.request(step.seqs[0]);
    ASSERT_GT(r.output.size(), 0u);
    const PagedBatch b = build_paged_batch(step, s, kMaxBlocks);
    ASSERT_EQ(b.num_tokens(), r.length());
    for (int64_t t = 0; t < b.num_tokens(); ++t) {
        EXPECT_EQ(b.token_ids[size_t(t)], r.token(t));
        EXPECT_EQ(b.positions[size_t(t)], t);
        EXPECT_EQ(b.context_lens[size_t(t)], t + 1);
    }
    EXPECT_EQ(b.logit_rows, (std::vector<int32_t>{int32_t(r.length() - 1)}));
}

TEST(PagedBatch, SampleRowsCarryParamsAndTheNumberOfTokensGenerated) {
    Scheduler s(config());
    s.submit(make(1, 4, 5, 11));
    s.submit(make(2, 3, 5, 22));
    ASSERT_TRUE(s.schedule().prefill);
    Step step{false, {1, 2}};
    auto rows = sample_rows(step, s);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].step, 0u);
    EXPECT_EQ(rows[0].params.seed, 11u);
    EXPECT_EQ(rows[1].params.seed, 22u);
    s.update(std::vector<int32_t>{7, 8});
    rows = sample_rows(step, s);
    EXPECT_EQ(rows[0].step, 1u);
    EXPECT_EQ(rows[1].step, 1u);
    EXPECT_FLOAT_EQ(rows[0].params.temperature, 0.5f);
}

TEST(PagedBatch, RejectsBadArgumentsWithInvalidArgument) {
    Scheduler s(config());
    s.submit(make(1, 6, 4));
    const Step step = s.schedule();
    EXPECT_THROW(build_paged_batch(step, s, 0), std::invalid_argument);
    EXPECT_THROW(build_paged_batch(step, s, 1), std::invalid_argument);
    EXPECT_THROW(build_paged_batch(Step{true, {}}, s, kMaxBlocks), std::invalid_argument);
    EXPECT_THROW(build_paged_batch(Step{true, {99}}, s, kMaxBlocks), std::invalid_argument);
}

TEST(PagedBatch, PaddingADecodeBatchAddsRowsThatWriteNothingAndAttendToNothing) {
    Scheduler s(config());
    s.submit(make(1, 4, 5));
    s.submit(make(2, 5, 5));
    ASSERT_TRUE(s.schedule().prefill);
    s.update(std::vector<int32_t>{901, 902});
    const Step step = s.schedule();
    const PagedBatch b = build_paged_batch(step, s, kMaxBlocks);
    const PagedBatch p = pad_decode_batch(b, 4);

    EXPECT_FALSE(p.prefill);
    EXPECT_EQ(p.num_seqs, 4);
    EXPECT_EQ(p.max_blocks, kMaxBlocks);
    EXPECT_EQ(p.token_ids, (std::vector<int32_t>{901, 902, 0, 0}));
    EXPECT_EQ(p.positions, (std::vector<int32_t>{4, 5, 0, 0}));
    EXPECT_EQ(p.slot_mapping, (std::vector<int32_t>{b.slot_mapping[0], b.slot_mapping[1], -1, -1}));
    EXPECT_EQ(p.context_lens, (std::vector<int32_t>{5, 6, 0, 0}));
    EXPECT_EQ(p.logit_rows, (std::vector<int32_t>{0, 1, 2, 3}));
    ASSERT_EQ(p.block_tables.size(), size_t(4 * kMaxBlocks));
    EXPECT_TRUE(std::equal(b.block_tables.begin(), b.block_tables.end(), p.block_tables.begin()));
    for (size_t i = b.block_tables.size(); i < p.block_tables.size(); ++i)
        EXPECT_EQ(p.block_tables[i], 0);
    EXPECT_EQ(pad_decode_batch(b, 2).token_ids, b.token_ids);
}

TEST(PagedBatch, PaddingRejectsPrefillBatchesAndTooFewRows) {
    Scheduler s(config());
    s.submit(make(1, 6, 4));
    const Step prefill = s.schedule();
    const PagedBatch pre = build_paged_batch(prefill, s, kMaxBlocks);
    EXPECT_THROW(pad_decode_batch(pre, 8), std::invalid_argument);
    s.update(std::vector<int32_t>{901});
    const PagedBatch dec = build_paged_batch(s.schedule(), s, kMaxBlocks);
    EXPECT_THROW(pad_decode_batch(dec, 0), std::invalid_argument);
    PagedBatch odd = dec;
    odd.logit_rows[0] = 0;
    odd.token_ids.push_back(5);
    EXPECT_THROW(pad_decode_batch(odd, 4), std::invalid_argument);
}

TEST(PagedBatch, LongestContextIsTheLargestContextLengthOfAnyRow) {
    PagedBatch b;
    EXPECT_EQ(b.longest_context(), 0);
    b.context_lens = {5, 700, 0, 12};
    EXPECT_EQ(b.longest_context(), 700);
}

TEST(AttentionSplits, SplitsOnlyWhenTheGridIsSmallAndTheContextIsLong) {
    EXPECT_EQ(attention_splits(1, 32, 128), 1);
    EXPECT_EQ(attention_splits(1, 32, 511), 1);
    EXPECT_EQ(attention_splits(1, 32, 512), 2);
    EXPECT_EQ(attention_splits(1, 32, 1023), 2);
    EXPECT_EQ(attention_splits(1, 32, 1024), 4);
    EXPECT_EQ(attention_splits(1, 32, 2048), 8);
    EXPECT_EQ(attention_splits(1, 32, 8192), 8);
    EXPECT_EQ(attention_splits(2, 32, 8192), 4);
    EXPECT_EQ(attention_splits(3, 32, 8192), 2);
    EXPECT_EQ(attention_splits(4, 32, 8192), 2);
    EXPECT_EQ(attention_splits(5, 32, 8192), 1);
    EXPECT_EQ(attention_splits(16, 32, 8192), 1);
    EXPECT_EQ(attention_splits(1, 4, 1 << 20), 8);
}

TEST(AttentionSplits, TheSplitGridNeverExceedsTheScratchBound) {
    for (int64_t heads : {1, 4, 32, 48, 64, 256})
        for (int64_t rows = 1; rows <= 64; ++rows)
            for (int64_t context : {0, 256, 512, 2048, 100000}) {
                const int64_t splits = attention_splits(rows, heads, context);
                EXPECT_GE(splits, 1);
                EXPECT_TRUE(splits == 1 || rows * heads * splits <= engine::kSplitAttentionBlocks);
            }
}
