#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/sched/speculative.h"

using engine::BlockManager;
using engine::count_accepted;
using engine::propose_prompt_lookup;

namespace {

constexpr int kBlock = 16;

using Tokens = std::vector<int32_t>;

Tokens reference_lookup(const Tokens& context, int max_ngram, int max_draft) {
    const int size = int(context.size());
    for (int g = std::min(max_ngram, size - 1); g >= 1; --g) {
        for (int i = size - 1 - g; i >= 0; --i) {
            bool same = true;
            for (int j = 0; j < g; ++j) same = same && context[i + j] == context[size - g + j];
            if (!same) continue;
            const int first = i + g;
            return Tokens(context.begin() + first,
                          context.begin() + std::min(first + max_draft, size));
        }
    }
    return {};
}

TEST(PromptLookup, CopiesWhatFollowedTheLastTokens) {
    const Tokens context{1, 2, 3, 4, 5, 6, 7, 9, 9, 2, 3, 4};
    EXPECT_EQ(propose_prompt_lookup(context, 3, 2), (Tokens{5, 6}));
    EXPECT_EQ(propose_prompt_lookup(context, 3, 4), (Tokens{5, 6, 7, 9}));
    EXPECT_EQ(propose_prompt_lookup(context, 3, 1), (Tokens{5}));
}

TEST(PromptLookup, NoRepeatGivesNothing) {
    EXPECT_TRUE(propose_prompt_lookup(Tokens{1, 2, 3, 4, 5}, 3, 4).empty());
    EXPECT_TRUE(propose_prompt_lookup(Tokens{}, 3, 4).empty());
    EXPECT_TRUE(propose_prompt_lookup(Tokens{7}, 3, 4).empty());
    EXPECT_TRUE(propose_prompt_lookup(Tokens{7, 8}, 3, 4).empty());
}

TEST(PromptLookup, TheLongestPatternWins) {
    const Tokens context{1, 2, 3, 50, 9, 3, 60, 1, 2, 3};
    EXPECT_EQ(propose_prompt_lookup(context, 3, 1), (Tokens{50}));
    EXPECT_EQ(propose_prompt_lookup(context, 2, 1), (Tokens{50}));
    EXPECT_EQ(propose_prompt_lookup(context, 1, 1), (Tokens{60}));
}

TEST(PromptLookup, FallsBackToShorterPatterns) {
    const Tokens context{4, 5, 6, 70, 8, 6};
    EXPECT_EQ(propose_prompt_lookup(context, 3, 2), (Tokens{70, 8}));
    const Tokens two{1, 5, 6, 70, 2, 5, 6};
    EXPECT_EQ(propose_prompt_lookup(two, 3, 2), (Tokens{70, 2}));
}

TEST(PromptLookup, TheLatestMatchWins) {
    const Tokens context{1, 2, 30, 0, 1, 2, 40, 0, 1, 2};
    EXPECT_EQ(propose_prompt_lookup(context, 2, 1), (Tokens{40}));
    EXPECT_EQ(propose_prompt_lookup(context, 2, 3), (Tokens{40, 0, 1}));
}

TEST(PromptLookup, TheDraftStopsAtTheEndOfTheContext) {
    const Tokens context{1, 2, 3, 4, 1, 2};
    EXPECT_EQ(propose_prompt_lookup(context, 2, 8), (Tokens{3, 4, 1, 2}));
    EXPECT_EQ(propose_prompt_lookup(context, 2, 4), (Tokens{3, 4, 1, 2}));
    EXPECT_EQ(propose_prompt_lookup(context, 2, 3), (Tokens{3, 4, 1}));
}

TEST(PromptLookup, AMatchMayOverlapThePattern) {
    EXPECT_EQ(propose_prompt_lookup(Tokens{5, 5, 5, 5}, 3, 4), (Tokens{5}));
    EXPECT_EQ(propose_prompt_lookup(Tokens{5, 5}, 3, 4), (Tokens{5}));
    EXPECT_EQ(propose_prompt_lookup(Tokens{1, 2, 1, 2, 1, 2}, 3, 8), (Tokens{1, 2}));
    EXPECT_EQ(propose_prompt_lookup(Tokens{1, 2, 1, 2, 1, 2}, 4, 8), (Tokens{1, 2}));
}

TEST(PromptLookup, ThePatternItselfIsNotAMatch) {
    EXPECT_TRUE(propose_prompt_lookup(Tokens{1, 2, 3, 8, 9}, 2, 4).empty());
    EXPECT_EQ(propose_prompt_lookup(Tokens{9, 1, 2, 3, 8, 9}, 2, 4), (Tokens{1, 2, 3, 8}));
}

TEST(PromptLookup, MaxNgramAboveTheContextLength) {
    const Tokens context{1, 2, 1};
    EXPECT_EQ(propose_prompt_lookup(context, 100, 5), (Tokens{2, 1}));
}

TEST(PromptLookup, RejectsBadArguments) {
    const Tokens context{1, 2, 1};
    EXPECT_THROW(propose_prompt_lookup(context, 0, 4), std::invalid_argument);
    EXPECT_THROW(propose_prompt_lookup(context, -1, 4), std::invalid_argument);
    EXPECT_THROW(propose_prompt_lookup(context, 3, 0), std::invalid_argument);
    EXPECT_THROW(propose_prompt_lookup(Tokens{}, 3, -2), std::invalid_argument);
    try {
        propose_prompt_lookup(context, 0, 4);
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("propose_prompt_lookup"), std::string::npos);
    }
}

TEST(PromptLookup, MatchesAPlainSearchOnRandomContexts) {
    std::mt19937 rng(102);
    int non_empty = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const int vocab = 2 + int(rng() % 5);
        Tokens context(rng() % 40);
        for (int32_t& token : context) token = int32_t(rng() % vocab);
        const int max_ngram = 1 + int(rng() % 6);
        const int max_draft = 1 + int(rng() % 10);
        const Tokens draft = propose_prompt_lookup(context, max_ngram, max_draft);
        ASSERT_EQ(draft, reference_lookup(context, max_ngram, max_draft))
            << "trial " << trial << " max_ngram " << max_ngram << " max_draft " << max_draft;
        ASSERT_LE(int(draft.size()), max_draft);
        if (!draft.empty()) non_empty++;
    }
    EXPECT_GT(non_empty, 2000);
}

TEST(PromptLookup, DoesNotChangeTheContextAndRepeats) {
    const Tokens context{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 1, 4};
    const Tokens copy = context;
    const Tokens first = propose_prompt_lookup(context, 3, 5);
    EXPECT_EQ(context, copy);
    EXPECT_EQ(first, (Tokens{1, 5, 9, 2, 6}));
    EXPECT_EQ(propose_prompt_lookup(context, 3, 5), first);
}

TEST(CountAccepted, CountsTheMatchingStart) {
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{1, 2, 3, 4}), 3);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{1, 2, 9, 4}), 2);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{9, 2, 3, 4}), 0);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{1, 9, 3, 4}), 1);
}

TEST(CountAccepted, ALaterMatchDoesNotCount) {
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3, 4}, Tokens{1, 7, 3, 4, 5}), 1);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3, 4}, Tokens{7, 2, 3, 4, 5}), 0);
}

TEST(CountAccepted, StopsAtTheShorterSpan) {
    EXPECT_EQ(count_accepted(Tokens{}, Tokens{5}), 0);
    EXPECT_EQ(count_accepted(Tokens{}, Tokens{}), 0);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{}), 0);
    EXPECT_EQ(count_accepted(Tokens{1, 2, 3}, Tokens{1, 2}), 2);
    EXPECT_EQ(count_accepted(Tokens{1, 2}, Tokens{1, 2, 3, 4, 5}), 2);
}

TEST(Truncate, GivesBackWholeBlocks) {
    BlockManager bm(8, kBlock);
    bm.allocate(1, 50);
    const std::vector<int> table = bm.table(1);
    ASSERT_EQ(table.size(), 4u);
    EXPECT_EQ(bm.num_free(), 3);
    bm.truncate(1, 33);
    EXPECT_EQ(bm.length(1), 33);
    EXPECT_EQ(bm.table(1), (std::vector<int>{table[0], table[1], table[2]}));
    EXPECT_EQ(bm.num_free(), 4);
    bm.truncate(1, 32);
    EXPECT_EQ(bm.table(1), (std::vector<int>{table[0], table[1]}));
    EXPECT_EQ(bm.num_free(), 5);
    bm.truncate(1, 1);
    EXPECT_EQ(bm.table(1), (std::vector<int>{table[0]}));
    EXPECT_EQ(bm.length(1), 1);
    EXPECT_EQ(bm.num_free(), 6);
}

TEST(Truncate, InsideABlockKeepsTheBlock) {
    BlockManager bm(8, kBlock);
    bm.allocate(1, 30);
    const std::vector<int> table = bm.table(1);
    bm.truncate(1, 20);
    EXPECT_EQ(bm.length(1), 20);
    EXPECT_EQ(bm.table(1), table);
    EXPECT_EQ(bm.num_free(), 5);
    EXPECT_EQ(bm.slot(1, 19), table[1] * kBlock + 3);
    EXPECT_THROW(bm.slot(1, 20), std::invalid_argument);
}

TEST(Truncate, ToTheCurrentLengthDoesNothing) {
    BlockManager bm(8, kBlock);
    bm.allocate(1, 32);
    const std::vector<int> table = bm.table(1);
    bm.truncate(1, 32);
    EXPECT_EQ(bm.length(1), 32);
    EXPECT_EQ(bm.table(1), table);
    EXPECT_EQ(bm.num_free(), 5);
}

TEST(Truncate, AppendContinuesAtTheNewLength) {
    BlockManager bm(8, kBlock);
    bm.allocate(1, 14);
    std::vector<int> slots;
    for (int i = 0; i < 6; ++i) slots.push_back(bm.append_slot(1));
    EXPECT_EQ(bm.length(1), 20);
    bm.truncate(1, 15);
    EXPECT_EQ(bm.table(1).size(), 1u);
    EXPECT_EQ(bm.append_slot(1), slots[1]);
    EXPECT_EQ(bm.length(1), 16);
    const int next = bm.append_slot(1);
    EXPECT_EQ(next % kBlock, 0);
    EXPECT_EQ(bm.table(1).size(), 2u);
    EXPECT_EQ(next, bm.table(1)[1] * kBlock);
}

TEST(Truncate, ReleasedBlocksCanBeAllocatedAgain) {
    BlockManager bm(4, kBlock);
    bm.allocate(1, 48);
    EXPECT_FALSE(bm.can_allocate(1));
    const std::vector<int> table = bm.table(1);
    bm.truncate(1, 16);
    EXPECT_TRUE(bm.can_allocate(32));
    EXPECT_FALSE(bm.can_allocate(33));
    bm.allocate(2, 32);
    const std::set<int> released{table[1], table[2]};
    EXPECT_EQ(std::set<int>(bm.table(2).begin(), bm.table(2).end()), released);
    EXPECT_FALSE(bm.can_append(1));
    bm.free(2);
    EXPECT_TRUE(bm.can_append(1));
}

TEST(Truncate, LeavesOtherSequencesAlone) {
    BlockManager bm(12, kBlock);
    bm.allocate(1, 40);
    bm.allocate(2, 40);
    bm.allocate(3, 40);
    const std::vector<int> first = bm.table(1), third = bm.table(3);
    bm.truncate(2, 5);
    EXPECT_EQ(bm.table(1), first);
    EXPECT_EQ(bm.table(3), third);
    EXPECT_EQ(bm.length(1), 40);
    EXPECT_EQ(bm.length(3), 40);
    EXPECT_EQ(bm.num_free(), 4);
    bm.free(2);
    EXPECT_EQ(bm.num_free(), 5);
}

TEST(Truncate, ASpeculativeStepWithEveryAcceptCount) {
    for (int accepted = 0; accepted <= 8; ++accepted) {
        SCOPED_TRACE(accepted);
        BlockManager bm(8, kBlock);
        const int n = 30;
        bm.allocate(1, n - 1);
        const int free_before = bm.num_free();
        std::vector<int> slots;
        for (int row = 0; row < 9; ++row) slots.push_back(bm.append_slot(1));
        EXPECT_EQ(bm.length(1), n + 8);
        bm.truncate(1, n + accepted);
        EXPECT_EQ(bm.length(1), n + accepted);
        EXPECT_EQ(bm.slot(1, n - 1 + accepted), slots[accepted]);
        const int blocks = (n + accepted + kBlock - 1) / kBlock;
        EXPECT_EQ(int(bm.table(1).size()), blocks);
        EXPECT_EQ(bm.num_free(), free_before - (blocks - 2));
        if (accepted < 8) {
            EXPECT_EQ(bm.append_slot(1), slots[accepted + 1]);
        }
    }
}

TEST(Truncate, OtherBlockSizes) {
    for (const int block_size : {1, 3, 32}) {
        SCOPED_TRACE(block_size);
        BlockManager bm(40, block_size);
        bm.allocate(1, 3 * block_size + 1);
        for (int length = 3 * block_size + 1; length >= 1; --length) {
            bm.truncate(1, length);
            ASSERT_EQ(bm.length(1), length);
            ASSERT_EQ(int(bm.table(1).size()), (length + block_size - 1) / block_size);
            ASSERT_EQ(bm.num_free(), 39 - int(bm.table(1).size()));
        }
    }
}

TEST(Truncate, RejectsBadArguments) {
    BlockManager bm(8, kBlock);
    bm.allocate(1, 20);
    EXPECT_THROW(bm.truncate(2, 5), std::invalid_argument);
    EXPECT_THROW(bm.truncate(1, 0), std::invalid_argument);
    EXPECT_THROW(bm.truncate(1, -3), std::invalid_argument);
    EXPECT_THROW(bm.truncate(1, 21), std::invalid_argument);
    EXPECT_EQ(bm.length(1), 20);
    EXPECT_EQ(bm.table(1).size(), 2u);
    EXPECT_EQ(bm.num_free(), 5);
    try {
        bm.truncate(1, 21);
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()).rfind("[BlockManager]", 0), 0u);
    }
}

}  // namespace
