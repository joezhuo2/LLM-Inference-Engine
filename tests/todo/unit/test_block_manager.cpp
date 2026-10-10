#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "engine/kv/block_manager.h"

using engine::BlockManager;
using engine::SeqId;

namespace {

constexpr int kBlock = 16;

int blocks_for(int tokens, int block_size = kBlock) {
    return (tokens + block_size - 1) / block_size;
}

struct Snapshot {
    int free = 0;
    std::map<SeqId, std::vector<int>> tables;
    std::map<SeqId, int> lengths;

    bool operator==(const Snapshot&) const = default;
};

Snapshot snapshot(const BlockManager& bm, const std::vector<SeqId>& seqs) {
    Snapshot s{bm.num_free(), {}, {}};
    for (const SeqId seq : seqs) {
        s.tables[seq] = bm.table(seq);
        s.lengths[seq] = bm.length(seq);
    }
    return s;
}

void expect_consistent(const BlockManager& bm, const std::map<SeqId, int>& lengths, int num_blocks,
                       int block_size = kBlock) {
    std::set<int> owned;
    std::set<int> slots;
    int total = 0;
    for (const auto& [seq, len] : lengths) {
        ASSERT_EQ(bm.length(seq), len) << "seq " << seq;
        const auto& table = bm.table(seq);
        ASSERT_EQ(static_cast<int>(table.size()), blocks_for(len, block_size)) << "seq " << seq;
        for (const int block : table) {
            ASSERT_GE(block, 1) << "seq " << seq;
            ASSERT_LT(block, num_blocks) << "seq " << seq;
            ASSERT_TRUE(owned.insert(block).second) << "block " << block << " owned twice";
        }
        total += static_cast<int>(table.size());
        for (int pos = 0; pos < len; ++pos) {
            const int slot = bm.slot(seq, pos);
            ASSERT_EQ(slot, table[pos / block_size] * block_size + pos % block_size)
                << "seq " << seq << " pos " << pos;
            ASSERT_TRUE(slots.insert(slot).second) << "slot " << slot << " used twice";
        }
    }
    ASSERT_EQ(bm.num_free() + total, num_blocks - 1);
}

}  // namespace

TEST(BlockManager, RejectsBadSizes) {
    EXPECT_THROW(BlockManager(0, kBlock), std::invalid_argument);
    EXPECT_THROW(BlockManager(1, kBlock), std::invalid_argument);
    EXPECT_THROW(BlockManager(-3, kBlock), std::invalid_argument);
    EXPECT_THROW(BlockManager(8, 0), std::invalid_argument);
    EXPECT_THROW(BlockManager(8, -1), std::invalid_argument);
}

TEST(BlockManager, StartsWithEveryBlockButTheNullBlockFree) {
    EXPECT_EQ(BlockManager(2, kBlock).num_free(), 1);
    EXPECT_EQ(BlockManager(9, kBlock).num_free(), 8);
    EXPECT_EQ(BlockManager(9, 1).num_free(), 8);
}

TEST(BlockManager, AllocateTakesOneBlockPerStartedBlockOfTokens) {
    const int num_blocks = 1 + 128 + 16;
    BlockManager bm(num_blocks, kBlock);
    std::map<SeqId, int> lengths;
    int expected_free = num_blocks - 1;
    SeqId seq = 0;
    for (const int len : {1, 15, 16, 17, 32, 33, 2048}) {
        bm.allocate(seq, len);
        lengths[seq++] = len;
        expected_free -= blocks_for(len);
        EXPECT_EQ(bm.num_free(), expected_free) << "length " << len;
    }
    EXPECT_EQ(bm.table(1).size(), 1u);
    EXPECT_EQ(bm.table(2).size(), 1u);
    EXPECT_EQ(bm.table(3).size(), 2u);
    EXPECT_EQ(bm.table(6).size(), 128u);
    expect_consistent(bm, lengths, num_blocks);
}

TEST(BlockManager, NeverHandsOutTheNullBlock) {
    const int num_blocks = 33;
    BlockManager bm(num_blocks, 4);
    std::map<SeqId, int> lengths;
    for (SeqId seq = 0; seq < 32; ++seq) {
        bm.allocate(seq, 3);
        lengths[seq] = 3;
    }
    EXPECT_EQ(bm.num_free(), 0);
    expect_consistent(bm, lengths, num_blocks, 4);
}

TEST(BlockManager, CanAllocateStopsAtTheFreeBlockCount) {
    BlockManager bm(4, kBlock);
    EXPECT_TRUE(bm.can_allocate(1));
    EXPECT_TRUE(bm.can_allocate(48));
    EXPECT_FALSE(bm.can_allocate(49));
    bm.allocate(7, 17);
    EXPECT_TRUE(bm.can_allocate(16));
    EXPECT_FALSE(bm.can_allocate(17));
    bm.allocate(8, 16);
    EXPECT_EQ(bm.num_free(), 0);
    EXPECT_FALSE(bm.can_allocate(1));
}

TEST(BlockManager, SlotFollowsTheBlockTable) {
    const int num_blocks = 12;
    BlockManager bm(num_blocks, kBlock);
    bm.allocate(1, 40);
    bm.allocate(2, 16);
    bm.allocate(3, 1);
    bm.free(2);
    bm.allocate(4, 50);
    expect_consistent(bm, {{1, 40}, {3, 1}, {4, 50}}, num_blocks);
}

TEST(BlockManager, AppendTakesABlockOnlyAtAMultipleOfTheBlockSize) {
    const int num_blocks = 8;
    BlockManager bm(num_blocks, kBlock);
    bm.allocate(5, 15);
    const int free = bm.num_free();

    const int slot15 = bm.append_slot(5);
    EXPECT_EQ(slot15, bm.slot(5, 15));
    EXPECT_EQ(bm.length(5), 16);
    EXPECT_EQ(bm.num_free(), free);
    EXPECT_EQ(bm.table(5).size(), 1u);

    const int slot16 = bm.append_slot(5);
    EXPECT_EQ(slot16, bm.slot(5, 16));
    EXPECT_EQ(bm.length(5), 17);
    EXPECT_EQ(bm.num_free(), free - 1);
    EXPECT_EQ(bm.table(5).size(), 2u);

    const int slot17 = bm.append_slot(5);
    EXPECT_EQ(slot17, bm.slot(5, 17));
    EXPECT_EQ(bm.num_free(), free - 1);
    expect_consistent(bm, {{5, 18}}, num_blocks);
}

TEST(BlockManager, AppendGrowsAOneTokenSequenceThroughManyBlocks) {
    const int num_blocks = 1 + 128;
    BlockManager bm(num_blocks, kBlock);
    bm.allocate(0, 1);
    for (int len = 1; len < 2048; ++len) {
        const int slot = bm.append_slot(0);
        ASSERT_EQ(slot, bm.slot(0, len));
        ASSERT_EQ(static_cast<int>(bm.table(0).size()), blocks_for(len + 1));
    }
    EXPECT_EQ(bm.num_free(), 0);
    expect_consistent(bm, {{0, 2048}}, num_blocks);
}

TEST(BlockManager, AFullCacheStillAppendsIntoPartlyFilledBlocks) {
    BlockManager bm(4, kBlock);
    bm.allocate(1, 20);
    bm.allocate(2, 16);
    EXPECT_EQ(bm.num_free(), 0);

    EXPECT_TRUE(bm.can_append(1));
    EXPECT_FALSE(bm.can_append(2));
    for (int len = 20; len < 32; ++len) bm.append_slot(1);
    EXPECT_FALSE(bm.can_append(1));
    EXPECT_EQ(bm.num_free(), 0);

    bm.free(2);
    EXPECT_TRUE(bm.can_append(1));
    bm.append_slot(1);
    expect_consistent(bm, {{1, 33}}, 4);
}

TEST(BlockManager, FreeReturnsEveryBlockAndForgetsTheSequence) {
    const int num_blocks = 10;
    BlockManager bm(num_blocks, kBlock);
    bm.allocate(1, 33);
    bm.allocate(2, 5);
    bm.append_slot(1);
    bm.free(1);
    EXPECT_EQ(bm.num_free(), num_blocks - 2);
    EXPECT_THROW(bm.length(1), std::invalid_argument);
    EXPECT_THROW(bm.table(1), std::invalid_argument);

    bm.allocate(1, 100);
    expect_consistent(bm, {{1, 100}, {2, 5}}, num_blocks);
    bm.free(2);
    bm.free(1);
    EXPECT_EQ(bm.num_free(), num_blocks - 1);
}

TEST(BlockManager, AcceptsAnyCallerChosenId) {
    BlockManager bm(4, kBlock);
    const SeqId big = INT64_C(1) << 40;
    bm.allocate(big, 3);
    bm.allocate(-1, 3);
    expect_consistent(bm, {{big, 3}, {-1, 3}}, 4);
}

TEST(BlockManager, RejectedCallsChangeNothing) {
    BlockManager bm(4, kBlock);
    bm.allocate(1, 16);
    bm.allocate(2, 30);
    const std::vector<SeqId> seqs{1, 2};
    const Snapshot before = snapshot(bm, seqs);

    EXPECT_THROW(bm.can_allocate(0), std::invalid_argument);
    EXPECT_THROW(bm.can_allocate(-5), std::invalid_argument);
    EXPECT_THROW(bm.allocate(3, 0), std::invalid_argument);
    EXPECT_THROW(bm.allocate(3, -1), std::invalid_argument);
    EXPECT_THROW(bm.allocate(1, 1), std::invalid_argument);
    EXPECT_THROW(bm.allocate(3, 1), std::invalid_argument);
    EXPECT_THROW(bm.append_slot(1), std::invalid_argument);
    EXPECT_THROW(bm.append_slot(3), std::invalid_argument);
    EXPECT_THROW(bm.can_append(3), std::invalid_argument);
    EXPECT_THROW(bm.free(3), std::invalid_argument);
    EXPECT_THROW(bm.slot(3, 0), std::invalid_argument);
    EXPECT_THROW(bm.slot(1, -1), std::invalid_argument);
    EXPECT_THROW(bm.slot(1, 16), std::invalid_argument);
    EXPECT_THROW(bm.table(3), std::invalid_argument);
    EXPECT_THROW(bm.length(3), std::invalid_argument);

    EXPECT_EQ(snapshot(bm, seqs), before);
    EXPECT_FALSE(bm.can_allocate(1));
    bm.free(2);
    bm.allocate(3, 32);
    EXPECT_EQ(bm.num_free(), 0);
}

TEST(BlockManager, AllocateFailsWholeWhenOnlySomeBlocksAreFree) {
    BlockManager bm(4, kBlock);
    bm.allocate(1, 16);
    EXPECT_THROW(bm.allocate(2, 33), std::invalid_argument);
    EXPECT_EQ(bm.num_free(), 2);
    EXPECT_THROW(bm.length(2), std::invalid_argument);
    bm.allocate(2, 32);
    expect_consistent(bm, {{1, 16}, {2, 32}}, 4);
}

TEST(BlockManager, SameCallsGiveSameBlocks) {
    auto run = [] {
        BlockManager bm(20, 4);
        bm.allocate(1, 9);
        bm.allocate(2, 4);
        bm.allocate(3, 13);
        bm.free(2);
        bm.append_slot(1);
        bm.append_slot(1);
        bm.append_slot(1);
        bm.append_slot(1);
        bm.allocate(4, 6);
        bm.free(1);
        bm.allocate(5, 21);
        return snapshot(bm, {3, 4, 5});
    };
    EXPECT_EQ(run(), run());
}

TEST(BlockManager, RandomOperationsKeepEveryInvariant) {
    const int num_blocks = 64;
    const int block_size = 4;
    BlockManager bm(num_blocks, block_size);
    std::map<SeqId, int> lengths;
    std::mt19937 rng(20261009);
    SeqId next = 0;
    int allocs = 0;
    int appends = 0;
    int frees = 0;
    int refused = 0;

    for (int step = 0; step < 10000; ++step) {
        const int op = std::uniform_int_distribution<int>(0, 9)(rng);
        if (op < 3 || lengths.empty()) {
            const int len = std::uniform_int_distribution<int>(1, 40)(rng);
            const bool fits = blocks_for(len, block_size) <= bm.num_free();
            ASSERT_EQ(bm.can_allocate(len), fits) << "step " << step;
            if (fits) {
                bm.allocate(next, len);
                lengths[next++] = len;
                ++allocs;
            } else {
                ASSERT_THROW(bm.allocate(next, len), std::invalid_argument) << "step " << step;
                ++refused;
            }
        } else {
            auto it = lengths.begin();
            std::advance(it, std::uniform_int_distribution<size_t>(0, lengths.size() - 1)(rng));
            const auto [seq, len] = *it;
            if (op < 8) {
                const bool fits = len % block_size != 0 || bm.num_free() > 0;
                ASSERT_EQ(bm.can_append(seq), fits) << "step " << step;
                if (fits) {
                    const int slot = bm.append_slot(seq);
                    it->second = len + 1;
                    ASSERT_EQ(slot, bm.slot(seq, len)) << "step " << step;
                    ++appends;
                } else {
                    ASSERT_THROW(bm.append_slot(seq), std::invalid_argument) << "step " << step;
                    ++refused;
                }
            } else {
                bm.free(seq);
                lengths.erase(it);
                ++frees;
            }
        }
        ASSERT_NO_FATAL_FAILURE(expect_consistent(bm, lengths, num_blocks, block_size))
            << "step " << step;
    }
    EXPECT_GT(allocs, 1000);
    EXPECT_GT(appends, 1000);
    EXPECT_GT(frees, 1000);
    EXPECT_GT(refused, 100);
}
