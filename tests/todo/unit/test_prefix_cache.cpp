#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kv/block_manager.h"

using engine::BlockManager;
using engine::SeqId;

namespace {

constexpr int kBlock = 16;

using Tokens = std::vector<int32_t>;

Tokens run(int count, int32_t first) {
    Tokens tokens(count);
    for (int i = 0; i < count; ++i) tokens[i] = first + i;
    return tokens;
}

Tokens head(const Tokens& tokens, int count) {
    return Tokens(tokens.begin(), tokens.begin() + count);
}

Tokens join(Tokens a, const Tokens& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::vector<int> first_blocks(const BlockManager& bm, SeqId seq, int count) {
    const auto& table = bm.table(seq);
    return std::vector<int>(table.begin(), table.begin() + count);
}

void hold_and_release(BlockManager& bm, SeqId seq, const Tokens& tokens) {
    bm.allocate_tokens(seq, tokens);
    bm.publish(seq, tokens);
    bm.free(seq);
}

TEST(PrefixCache, NothingPublishedMatchesNothing) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_TRUE(bm.can_allocate_tokens(a));
    EXPECT_EQ(bm.allocate_tokens(1, a), 0);
    EXPECT_EQ(bm.length(1), 40);
    EXPECT_EQ(bm.table(1), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(bm.num_free(), 4);
    EXPECT_EQ(bm.num_cached(), 0);
    for (const int block : bm.table(1)) EXPECT_EQ(bm.ref_count(block), 1);
    EXPECT_EQ(bm.ref_count(0), 0);
    EXPECT_EQ(bm.ref_count(4), 0);
    EXPECT_EQ(bm.match_prefix(a), 0);
}

TEST(PrefixCache, AllocateTokensWithoutHitsActsLikeAllocate) {
    BlockManager by_count(40, kBlock);
    BlockManager by_tokens(40, kBlock);
    const std::vector<int> lengths{1, 15, 16, 17, 32, 33, 100};
    for (size_t i = 0; i < lengths.size(); ++i) {
        const SeqId seq = SeqId(i);
        by_count.allocate(seq, lengths[i]);
        EXPECT_EQ(by_tokens.allocate_tokens(seq, run(lengths[i], int32_t(1000 * (i + 1)))), 0);
        EXPECT_EQ(by_tokens.table(seq), by_count.table(seq));
        EXPECT_EQ(by_tokens.length(seq), by_count.length(seq));
        EXPECT_EQ(by_tokens.num_free(), by_count.num_free());
    }
    by_count.free(2);
    by_tokens.free(2);
    EXPECT_EQ(by_tokens.num_free(), by_count.num_free());
    EXPECT_EQ(by_tokens.num_cached(), 0);
    by_count.allocate(50, 20);
    by_tokens.allocate_tokens(50, run(20, 9000));
    EXPECT_EQ(by_tokens.table(50), by_count.table(50));
}

TEST(PrefixCache, PublishMakesFullBlocksMatch) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_EQ(bm.match_prefix(join(a, run(30, 500))), 32);
    EXPECT_EQ(bm.match_prefix(head(a, 33)), 32);
    EXPECT_EQ(bm.match_prefix(head(a, 17)), 16);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.num_free(), 4);
}

TEST(PrefixCache, TheLastTokenIsNeverMatched) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(48, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_EQ(bm.match_prefix(join(a, run(1, 7))), 48);
    EXPECT_EQ(bm.match_prefix(head(a, 32)), 16);
    EXPECT_EQ(bm.match_prefix(head(a, 16)), 0);
    EXPECT_EQ(bm.match_prefix(head(a, 1)), 0);
    EXPECT_EQ(bm.allocate_tokens(2, head(a, 32)), 16);
    EXPECT_EQ(bm.table(2)[0], bm.table(1)[0]);
    EXPECT_NE(bm.table(2)[1], bm.table(1)[1]);
}

TEST(PrefixCache, PublishCoversOnlyTheTokensItIsGiven) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, {});
    EXPECT_EQ(bm.match_prefix(a), 0);
    bm.publish(1, head(a, 15));
    EXPECT_EQ(bm.match_prefix(a), 0);
    bm.publish(1, head(a, 20));
    EXPECT_EQ(bm.match_prefix(a), 16);
    bm.publish(1, head(a, 35));
    EXPECT_EQ(bm.match_prefix(a), 32);
    bm.publish(1, head(a, 20));
    EXPECT_EQ(bm.match_prefix(a), 32);
}

TEST(PrefixCache, MatchingFollowsTheWholeChain) {
    BlockManager bm(12, kBlock);
    const Tokens a0 = run(16, 100), a1 = run(16, 200), a2 = run(16, 300);
    const Tokens a = join(join(a0, a1), a2);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    EXPECT_EQ(bm.match_prefix(join(a, run(1, 7))), 48);
    EXPECT_EQ(bm.match_prefix(join(join(run(16, 900), a1), a2)), 0);
    EXPECT_EQ(bm.match_prefix(join(join(a0, run(16, 900)), a2)), 16);
    EXPECT_EQ(bm.match_prefix(join(join(a1, a0), a2)), 0);
    Tokens off_by_one = join(a, run(1, 7));
    off_by_one[31] += 1;
    EXPECT_EQ(bm.match_prefix(off_by_one), 16);
    off_by_one[0] += 1;
    EXPECT_EQ(bm.match_prefix(off_by_one), 0);
}

TEST(PrefixCache, AllocateTokensSharesMatchedBlocks) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    const Tokens b = join(head(a, 32), run(20, 500));
    EXPECT_TRUE(bm.can_allocate_tokens(b));
    EXPECT_EQ(bm.allocate_tokens(2, b), 32);
    EXPECT_EQ(bm.length(2), 52);
    ASSERT_EQ(bm.table(2).size(), 4u);
    EXPECT_EQ(first_blocks(bm, 2, 2), first_blocks(bm, 1, 2));
    EXPECT_EQ(bm.ref_count(bm.table(1)[0]), 2);
    EXPECT_EQ(bm.ref_count(bm.table(1)[1]), 2);
    EXPECT_EQ(bm.ref_count(bm.table(1)[2]), 1);
    const std::set<int> mine(bm.table(1).begin(), bm.table(1).end());
    for (int i = 2; i < 4; ++i) {
        EXPECT_EQ(mine.count(bm.table(2)[i]), 0u);
        EXPECT_EQ(bm.ref_count(bm.table(2)[i]), 1);
    }
    EXPECT_EQ(bm.num_free(), 2);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.slot(2, 5), bm.slot(1, 5));
    EXPECT_NE(bm.slot(2, 35), bm.slot(1, 35));
}

TEST(PrefixCache, SharedBlocksNeedNoFreeBlocks) {
    BlockManager bm(5, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    EXPECT_EQ(bm.num_free(), 1);
    const Tokens b = join(head(a, 32), run(10, 500));
    EXPECT_FALSE(bm.can_allocate(42));
    EXPECT_TRUE(bm.can_allocate_tokens(b));
    EXPECT_FALSE(bm.can_allocate_tokens(join(head(a, 32), run(17, 500))));
    EXPECT_FALSE(bm.can_allocate_tokens(run(42, 700)));
    EXPECT_EQ(bm.allocate_tokens(2, b), 32);
    EXPECT_EQ(bm.num_free(), 0);
    EXPECT_FALSE(bm.can_allocate_tokens(join(head(a, 32), run(1, 500))));
}

TEST(PrefixCache, AppendNeverWritesIntoASharedBlock) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(32, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    EXPECT_EQ(bm.allocate_tokens(2, join(a, run(1, 500))), 32);
    EXPECT_EQ(first_blocks(bm, 2, 2), bm.table(1));
    const int slot1 = bm.append_slot(1);
    const int slot2 = bm.append_slot(2);
    ASSERT_EQ(bm.table(1).size(), 3u);
    ASSERT_EQ(bm.table(2).size(), 3u);
    EXPECT_NE(bm.table(1)[2], bm.table(2)[2]);
    EXPECT_EQ(slot1, bm.table(1)[2] * kBlock);
    EXPECT_EQ(slot2, bm.table(2)[2] * kBlock + 1);
    EXPECT_EQ(bm.ref_count(bm.table(1)[2]), 1);
    EXPECT_EQ(bm.ref_count(bm.table(2)[2]), 1);
    EXPECT_EQ(bm.ref_count(bm.table(1)[0]), 2);
}

TEST(PrefixCache, FreeKeepsPublishedBlocks) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    const std::vector<int> blocks = first_blocks(bm, 1, 2);
    bm.free(1);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.num_free(), 7);
    EXPECT_EQ(bm.ref_count(blocks[0]), 0);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_EQ(bm.allocate_tokens(2, a), 32);
    EXPECT_EQ(first_blocks(bm, 2, 2), blocks);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.num_free(), 4);
    EXPECT_EQ(bm.ref_count(blocks[0]), 1);
    EXPECT_EQ(bm.ref_count(blocks[1]), 1);
}

TEST(PrefixCache, UnpublishedBlocksGoBlankOnFree) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.free(1);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.num_free(), 7);
    EXPECT_EQ(bm.match_prefix(a), 0);
    bm.allocate_tokens(2, a);
    bm.publish(2, head(a, 16));
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 1);
    EXPECT_EQ(bm.num_free(), 7);
    EXPECT_EQ(bm.match_prefix(a), 16);
}

TEST(PrefixCache, ASharedBlockStaysInUseUntilItsLastHolderIsFreed) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    bm.allocate_tokens(2, a);
    const std::vector<int> shared = first_blocks(bm, 1, 2);
    bm.free(1);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.ref_count(shared[0]), 1);
    EXPECT_EQ(bm.num_free(), 4);
    EXPECT_EQ(first_blocks(bm, 2, 2), shared);
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.num_free(), 7);
}

TEST(PrefixCache, BlankBlocksAreUsedBeforeACachedBlockIsEvicted) {
    BlockManager bm(6, kBlock);
    const Tokens a = run(33, 100);
    hold_and_release(bm, 1, a);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.num_free(), 5);
    bm.allocate(2, 48);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_EQ(bm.num_free(), 2);
    EXPECT_TRUE(bm.can_allocate(32));
    EXPECT_FALSE(bm.can_allocate(33));
}

TEST(PrefixCache, EvictionTakesTheEndOfAChainFirst) {
    BlockManager bm(6, kBlock);
    const Tokens a = run(33, 100);
    hold_and_release(bm, 1, a);
    bm.allocate(2, 48);
    bm.allocate(3, 16);
    EXPECT_EQ(bm.num_cached(), 1);
    EXPECT_EQ(bm.match_prefix(a), 16);
    EXPECT_TRUE(bm.can_append(3));
    bm.append_slot(3);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_EQ(bm.num_free(), 0);
    EXPECT_FALSE(bm.can_allocate(1));
    std::set<int> owned;
    for (const SeqId seq : {SeqId(2), SeqId(3)}) {
        for (const int block : bm.table(seq)) EXPECT_TRUE(owned.insert(block).second);
    }
    EXPECT_EQ(owned.size(), 5u);
}

TEST(PrefixCache, EvictionIsLeastRecentlyReleasedFirst) {
    BlockManager bm(7, kBlock);
    const Tokens a = run(33, 100), b = run(33, 500);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    bm.allocate_tokens(2, b);
    bm.publish(2, b);
    bm.free(1);
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 4);
    EXPECT_EQ(bm.num_free(), 6);
    bm.allocate(10, 48);
    EXPECT_EQ(bm.match_prefix(a), 16);
    EXPECT_EQ(bm.match_prefix(b), 32);
    bm.allocate(11, 16);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_EQ(bm.match_prefix(b), 32);
    bm.allocate(12, 16);
    EXPECT_EQ(bm.match_prefix(b), 16);
    EXPECT_EQ(bm.num_cached(), 1);
}

TEST(PrefixCache, AHitRefreshesTheBlocksItUsed) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(33, 100), b = run(33, 500);
    bm.allocate_tokens(1, a);
    bm.publish(1, a);
    bm.allocate_tokens(2, b);
    bm.publish(2, b);
    bm.free(1);
    bm.free(2);
    EXPECT_EQ(bm.allocate_tokens(3, a), 32);
    bm.free(3);
    EXPECT_EQ(bm.num_cached(), 4);
    EXPECT_EQ(bm.num_free(), 7);
    bm.allocate(10, 64);
    EXPECT_EQ(bm.match_prefix(b), 16);
    EXPECT_EQ(bm.match_prefix(a), 32);
    bm.allocate(11, 16);
    EXPECT_EQ(bm.match_prefix(b), 0);
    EXPECT_EQ(bm.match_prefix(a), 32);
}

TEST(PrefixCache, MatchPrefixDoesNotRefresh) {
    BlockManager bm(6, kBlock);
    const Tokens a = run(17, 100), b = run(17, 500);
    hold_and_release(bm, 1, a);
    hold_and_release(bm, 2, b);
    EXPECT_EQ(bm.match_prefix(a), 16);
    EXPECT_EQ(bm.num_free(), 5);
    bm.allocate(10, 64);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_EQ(bm.match_prefix(b), 16);
}

TEST(PrefixCache, EvictionSparesTheBlocksOfTheSameCall) {
    BlockManager bm(4, kBlock);
    const Tokens a = run(33, 100);
    hold_and_release(bm, 1, a);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.num_free(), 3);
    const Tokens too_long = join(head(a, 32), run(20, 500));
    EXPECT_FALSE(bm.can_allocate_tokens(too_long));
    EXPECT_THROW(bm.allocate_tokens(2, too_long), std::invalid_argument);
    EXPECT_EQ(bm.num_cached(), 2);
    EXPECT_EQ(bm.num_free(), 3);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_THROW(bm.length(2), std::invalid_argument);
    const Tokens fits = join(head(a, 32), run(16, 500));
    EXPECT_TRUE(bm.can_allocate_tokens(fits));
    EXPECT_EQ(bm.allocate_tokens(2, fits), 32);
    EXPECT_EQ(bm.num_free(), 0);
    EXPECT_EQ(bm.num_cached(), 0);
    const std::set<int> blocks(bm.table(2).begin(), bm.table(2).end());
    EXPECT_EQ(blocks.size(), 3u);
}

TEST(PrefixCache, AMissMayEvictEverything) {
    BlockManager bm(4, kBlock);
    const Tokens a = run(33, 100);
    hold_and_release(bm, 1, a);
    const Tokens other = run(48, 700);
    EXPECT_TRUE(bm.can_allocate_tokens(other));
    EXPECT_EQ(bm.allocate_tokens(2, other), 0);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.num_free(), 0);
    EXPECT_EQ(bm.match_prefix(a), 0);
    bm.publish(2, other);
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 3);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_EQ(bm.match_prefix(join(other, run(1, 7))), 48);
}

TEST(PrefixCache, EvictionSparesMatchedBlocksEvenWhenTheyAreOldest) {
    BlockManager bm(4, kBlock);
    const Tokens a = run(33, 100);
    const Tokens c = run(16, 500);
    hold_and_release(bm, 1, a);
    hold_and_release(bm, 2, c);
    EXPECT_EQ(bm.num_cached(), 3);
    EXPECT_EQ(bm.num_free(), 3);
    EXPECT_EQ(bm.match_prefix(join(c, run(1, 7))), 16);
    const Tokens next = join(head(a, 32), run(16, 900));
    EXPECT_TRUE(bm.can_allocate_tokens(next));
    EXPECT_EQ(bm.allocate_tokens(3, next), 32);
    EXPECT_EQ(bm.table(3), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(bm.match_prefix(join(c, run(1, 7))), 0);
    EXPECT_EQ(bm.num_cached(), 0);
    bm.free(3);
    EXPECT_EQ(bm.match_prefix(a), 32);
}

TEST(PrefixCache, AnEvictedBlockCanBePublishedForOtherTokens) {
    BlockManager bm(3, kBlock);
    const Tokens a = run(32, 100), b = run(32, 500);
    const Tokens tail = run(1, 7);
    hold_and_release(bm, 1, a);
    EXPECT_EQ(bm.match_prefix(join(a, tail)), 32);
    EXPECT_EQ(bm.allocate_tokens(2, b), 0);
    bm.publish(2, b);
    EXPECT_EQ(bm.match_prefix(join(a, tail)), 0);
    EXPECT_EQ(bm.match_prefix(join(b, tail)), 32);
    const Tokens a1(a.begin() + 16, a.end()), b1(b.begin() + 16, b.end());
    EXPECT_EQ(bm.match_prefix(join(join(head(b, 16), a1), tail)), 16);
    EXPECT_EQ(bm.match_prefix(join(join(head(a, 16), b1), tail)), 0);
    bm.free(2);
    EXPECT_EQ(bm.match_prefix(join(join(head(a, 16), b1), tail)), 0);
    EXPECT_EQ(bm.match_prefix(join(b, tail)), 32);
}

TEST(PrefixCache, TheFirstPublishedBlockWinsADuplicate) {
    BlockManager bm(12, kBlock);
    const Tokens a = run(40, 100);
    EXPECT_EQ(bm.allocate_tokens(1, a), 0);
    EXPECT_EQ(bm.allocate_tokens(2, a), 0);
    bm.publish(1, a);
    bm.publish(2, a);
    EXPECT_EQ(bm.match_prefix(a), 32);
    EXPECT_EQ(bm.allocate_tokens(3, a), 32);
    EXPECT_EQ(first_blocks(bm, 3, 2), first_blocks(bm, 1, 2));
    EXPECT_EQ(bm.ref_count(bm.table(2)[0]), 1);
    EXPECT_EQ(bm.ref_count(bm.table(2)[1]), 1);
    EXPECT_EQ(bm.ref_count(bm.table(1)[0]), 2);
    const int before = bm.num_free();
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 0);
    EXPECT_EQ(bm.num_free(), before + 3);
    EXPECT_EQ(bm.match_prefix(a), 32);
}

TEST(PrefixCache, GeneratedTokensCanBePublished) {
    BlockManager bm(12, kBlock);
    const Tokens prompt = run(20, 100);
    const Tokens generated = run(30, 500);
    bm.allocate_tokens(1, prompt);
    for (size_t i = 0; i < generated.size(); ++i) bm.append_slot(1);
    const Tokens turn = join(prompt, generated);
    bm.publish(1, turn);
    const Tokens next = join(turn, run(10, 900));
    EXPECT_EQ(bm.match_prefix(next), 48);
    EXPECT_EQ(bm.allocate_tokens(2, next), 48);
    EXPECT_EQ(first_blocks(bm, 2, 3), first_blocks(bm, 1, 3));
    EXPECT_NE(bm.table(2)[3], bm.table(1)[3]);
    bm.free(1);
    EXPECT_EQ(bm.num_cached(), 0);
    bm.free(2);
    EXPECT_EQ(bm.num_cached(), 3);
}

TEST(PrefixCache, BlocksFromPlainAllocateCanBePublished) {
    BlockManager bm(8, kBlock);
    const Tokens a = run(40, 100);
    bm.allocate(1, 40);
    EXPECT_EQ(bm.match_prefix(a), 0);
    bm.publish(1, a);
    EXPECT_EQ(bm.match_prefix(a), 32);
}

TEST(PrefixCache, OtherBlockSizes) {
    for (const int block_size : {1, 2, 5, 32}) {
        SCOPED_TRACE(block_size);
        for (const int n : {1, block_size, block_size + 1, 3 * block_size, 3 * block_size + 1}) {
            SCOPED_TRACE(n);
            BlockManager bm(64, block_size);
            const Tokens a = run(n, 100);
            EXPECT_EQ(bm.allocate_tokens(1, a), 0);
            bm.publish(1, a);
            EXPECT_EQ(bm.match_prefix(a), (n - 1) / block_size * block_size);
            EXPECT_EQ(bm.match_prefix(join(a, run(1, 7))), n / block_size * block_size);
            bm.free(1);
            EXPECT_EQ(bm.num_cached(), n / block_size);
            EXPECT_EQ(bm.num_free(), 63);
            EXPECT_EQ(bm.allocate_tokens(2, a), (n - 1) / block_size * block_size);
            EXPECT_EQ(bm.num_cached(), n / block_size - (n - 1) / block_size);
        }
    }
}

TEST(PrefixCache, RejectsBadArguments) {
    BlockManager bm(4, kBlock);
    const Tokens none;
    const Tokens a = run(20, 100);
    EXPECT_THROW(bm.match_prefix(none), std::invalid_argument);
    EXPECT_THROW(bm.can_allocate_tokens(none), std::invalid_argument);
    EXPECT_THROW(bm.allocate_tokens(1, none), std::invalid_argument);
    bm.allocate_tokens(1, a);
    EXPECT_THROW(bm.allocate_tokens(1, run(5, 500)), std::invalid_argument);
    EXPECT_THROW(bm.allocate_tokens(2, run(33, 500)), std::invalid_argument);
    EXPECT_THROW(bm.length(2), std::invalid_argument);
    EXPECT_EQ(bm.num_free(), 1);
    EXPECT_THROW(bm.publish(9, a), std::invalid_argument);
    EXPECT_THROW(bm.publish(1, run(21, 100)), std::invalid_argument);
    EXPECT_EQ(bm.match_prefix(a), 0);
    EXPECT_THROW(bm.ref_count(-1), std::invalid_argument);
    EXPECT_THROW(bm.ref_count(4), std::invalid_argument);
    EXPECT_EQ(bm.ref_count(3), 0);
    try {
        bm.publish(9, a);
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()).rfind("[BlockManager]", 0), 0u);
    }
}

TEST(PrefixCache, RandomCallsNeverMatchABlockHoldingOtherTokens) {
    for (const int block_size : {1, 2, 3}) {
        SCOPED_TRACE(block_size);
        const int num_blocks = 14;
        BlockManager bm(num_blocks, block_size);
        std::mt19937 rng(1234u + unsigned(block_size));
        std::map<SeqId, Tokens> live;
        std::map<int, Tokens> holds;
        SeqId next_id = 1;
        int hits = 0;

        const auto check = [&] {
            std::vector<int> refs(num_blocks, 0);
            for (const auto& [seq, tokens] : live) {
                const auto& table = bm.table(seq);
                ASSERT_EQ(bm.length(seq), int(tokens.size()));
                ASSERT_EQ(int(table.size()), (int(tokens.size()) + block_size - 1) / block_size);
                std::set<int> once;
                for (const int block : table) {
                    ASSERT_GE(block, 1);
                    ASSERT_LT(block, num_blocks);
                    ASSERT_TRUE(once.insert(block).second);
                    refs[block]++;
                }
                if (tokens.size() % block_size != 0) {
                    ASSERT_EQ(bm.ref_count(table.back()), 1);
                }
            }
            int free = 0;
            int cached = 0;
            for (int block = 1; block < num_blocks; ++block) {
                ASSERT_EQ(bm.ref_count(block), refs[block]) << "block " << block;
                if (refs[block] == 0) free++;
                if (refs[block] == 0 && holds.count(block)) cached++;
            }
            ASSERT_EQ(bm.num_free(), free);
            ASSERT_LE(bm.num_cached(), cached);
        };

        for (int step = 0; step < 6000; ++step) {
            SCOPED_TRACE(step);
            const int op = int(rng() % 10);
            if (op < 4) {
                const int n = 1 + int(rng() % (4 * block_size + 2));
                Tokens tokens(n);
                for (int& t : tokens) t = int32_t(rng() % 2);
                const int matched = bm.match_prefix(tokens);
                ASSERT_EQ(matched % block_size, 0);
                ASSERT_LE(matched, (n - 1) / block_size * block_size);
                int cached_hits = 0;
                const bool can = bm.can_allocate_tokens(tokens);
                const int before = bm.num_free();
                if (!can) {
                    ASSERT_THROW(bm.allocate_tokens(next_id, tokens), std::invalid_argument);
                    ASSERT_EQ(bm.num_free(), before);
                    continue;
                }
                std::vector<int> refs_before(num_blocks);
                for (int block = 0; block < num_blocks; ++block) {
                    refs_before[block] = bm.ref_count(block);
                }
                const SeqId seq = next_id++;
                ASSERT_EQ(bm.allocate_tokens(seq, tokens), matched);
                const auto table = bm.table(seq);
                const int shared = matched / block_size;
                for (int i = 0; i < int(table.size()); ++i) {
                    if (i < shared) {
                        ASSERT_TRUE(holds.count(table[i])) << "matched a block never published";
                        ASSERT_EQ(holds[table[i]], head(tokens, (i + 1) * block_size));
                        ASSERT_EQ(bm.ref_count(table[i]), refs_before[table[i]] + 1);
                        if (refs_before[table[i]] == 0) cached_hits++;
                    } else {
                        ASSERT_EQ(refs_before[table[i]], 0);
                        ASSERT_EQ(bm.ref_count(table[i]), 1);
                        holds.erase(table[i]);
                    }
                }
                ASSERT_EQ(bm.num_free(), before - cached_hits - (int(table.size()) - shared));
                hits += shared;
                live[seq] = tokens;
            } else if (op < 6 && !live.empty()) {
                auto it = std::next(live.begin(), rng() % live.size());
                if (!bm.can_append(it->first)) continue;
                const bool new_block = it->second.size() % block_size == 0;
                bm.append_slot(it->first);
                it->second.push_back(int32_t(rng() % 2));
                if (new_block) {
                    const int block = bm.table(it->first).back();
                    ASSERT_EQ(bm.ref_count(block), 1);
                    holds.erase(block);
                }
            } else if (op < 8 && !live.empty()) {
                auto it = std::next(live.begin(), rng() % live.size());
                const int count = int(rng() % (it->second.size() + 1));
                bm.publish(it->first, head(it->second, count));
                const auto& table = bm.table(it->first);
                for (int i = 0; i < count / block_size; ++i) {
                    const Tokens prefix = head(it->second, (i + 1) * block_size);
                    if (holds.count(table[i])) {
                        ASSERT_EQ(holds[table[i]], prefix);
                    }
                    holds[table[i]] = prefix;
                }
            } else if (!live.empty()) {
                auto it = std::next(live.begin(), rng() % live.size());
                bm.free(it->first);
                live.erase(it);
            }
            check();
            if (HasFatalFailure()) return;
        }
        EXPECT_GT(hits, 100);
    }
}

}  // namespace
