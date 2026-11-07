#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "engine/sched/graph_buckets.h"

using engine::decode_graph_buckets;

namespace {

using Buckets = std::vector<int64_t>;

Buckets powers_of_two(int64_t max_batch) {
    Buckets buckets;
    for (int64_t b = 1; b < max_batch; b *= 2) buckets.push_back(b);
    buckets.push_back(max_batch);
    return buckets;
}

int64_t bucket_for(int64_t seqs, const Buckets& buckets) {
    for (int64_t b : buckets) {
        if (seqs <= b) return b;
    }
    return 0;
}

bool contains(const Buckets& buckets, int64_t value) {
    return std::find(buckets.begin(), buckets.end(), value) != buckets.end();
}

constexpr int64_t kSteps[] = {1, 2, 4, 8, 16};

}  // namespace

TEST(GraphBuckets, OneStepPerOctaveIsTodaysGrid) {
    for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
        EXPECT_EQ(decode_graph_buckets(max_batch, 1), powers_of_two(max_batch)) << max_batch;
    }
}

TEST(GraphBuckets, OneStepPerOctaveExamples) {
    EXPECT_EQ(decode_graph_buckets(16, 1), (Buckets{1, 2, 4, 8, 16}));
    EXPECT_EQ(decode_graph_buckets(17, 1), (Buckets{1, 2, 4, 8, 16, 17}));
    EXPECT_EQ(decode_graph_buckets(256, 1), (Buckets{1, 2, 4, 8, 16, 32, 64, 128, 256}));
}

TEST(GraphBuckets, TwoStepsPerOctave) {
    EXPECT_EQ(decode_graph_buckets(64, 2), (Buckets{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}));
}

TEST(GraphBuckets, FourStepsPerOctave) {
    EXPECT_EQ(decode_graph_buckets(32, 4),
              (Buckets{1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}));
}

TEST(GraphBuckets, EightStepsPerOctaveTo64) {
    EXPECT_EQ(decode_graph_buckets(64, 8),
              (Buckets{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                       18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64}));
}

TEST(GraphBuckets, StepsAboveMaxBatchGiveEveryInteger) {
    EXPECT_EQ(decode_graph_buckets(5, 8), (Buckets{1, 2, 3, 4, 5}));
    EXPECT_EQ(decode_graph_buckets(16, 16),
              (Buckets{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
}

TEST(GraphBuckets, MaxBatchOneIsOneBucket) {
    for (int64_t steps : kSteps) {
        EXPECT_EQ(decode_graph_buckets(1, steps), (Buckets{1})) << steps;
    }
}

TEST(GraphBuckets, MaxBatchOnTheGridIsNotRepeated) {
    EXPECT_EQ(decode_graph_buckets(12, 2), (Buckets{1, 2, 3, 4, 6, 8, 12}));
    EXPECT_EQ(decode_graph_buckets(48, 2), (Buckets{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}));
    EXPECT_EQ(decode_graph_buckets(2, 1), (Buckets{1, 2}));
}

TEST(GraphBuckets, MaxBatchOffTheGridEndsTheList) {
    EXPECT_EQ(decode_graph_buckets(13, 2), (Buckets{1, 2, 3, 4, 6, 8, 12, 13}));
    EXPECT_EQ(decode_graph_buckets(20, 2), (Buckets{1, 2, 3, 4, 6, 8, 12, 16, 20}));
    EXPECT_EQ(decode_graph_buckets(33, 4),
              (Buckets{1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 33}));
    EXPECT_EQ(decode_graph_buckets(100, 2),
              (Buckets{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 100}));
}

TEST(GraphBuckets, BucketCountsTo256) {
    EXPECT_EQ(decode_graph_buckets(256, 1).size(), 9u);
    EXPECT_EQ(decode_graph_buckets(256, 2).size(), 16u);
    EXPECT_EQ(decode_graph_buckets(256, 4).size(), 28u);
    EXPECT_EQ(decode_graph_buckets(256, 8).size(), 48u);
}

TEST(GraphBuckets, StrictlyIncreasingFromOneToMaxBatch) {
    for (int64_t steps : kSteps) {
        for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
            const Buckets buckets = decode_graph_buckets(max_batch, steps);
            ASSERT_FALSE(buckets.empty());
            EXPECT_EQ(buckets.front(), 1) << max_batch << " " << steps;
            EXPECT_EQ(buckets.back(), max_batch) << max_batch << " " << steps;
            EXPECT_TRUE(std::adjacent_find(buckets.begin(), buckets.end(),
                                           std::greater_equal<>()) == buckets.end())
                << max_batch << " " << steps;
        }
    }
}

TEST(GraphBuckets, EveryPowerOfTwoIsABucket) {
    for (int64_t steps : kSteps) {
        for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
            const Buckets buckets = decode_graph_buckets(max_batch, steps);
            for (int64_t p = 1; p <= max_batch; p *= 2) {
                EXPECT_TRUE(contains(buckets, p)) << max_batch << " " << steps << " " << p;
            }
        }
    }
}

TEST(GraphBuckets, FinerGridContainsCoarser) {
    for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
        for (int64_t steps = 1; steps <= 8; steps *= 2) {
            const Buckets coarse = decode_graph_buckets(max_batch, steps);
            const Buckets fine = decode_graph_buckets(max_batch, steps * 2);
            EXPECT_TRUE(std::includes(fine.begin(), fine.end(), coarse.begin(), coarse.end()))
                << max_batch << " " << steps;
        }
    }
}

TEST(GraphBuckets, NeighbourGapIsBounded) {
    for (int64_t steps : kSteps) {
        for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
            const Buckets buckets = decode_graph_buckets(max_batch, steps);
            for (size_t i = 1; i < buckets.size(); ++i) {
                const int64_t gap = buckets[i] - buckets[i - 1];
                EXPECT_LE(gap, std::max<int64_t>(1, buckets[i - 1] / steps))
                    << max_batch << " " << steps << " " << buckets[i - 1];
            }
        }
    }
}

TEST(GraphBuckets, PaddingIsBelowSeqsOverSteps) {
    for (int64_t steps : kSteps) {
        for (int64_t max_batch = 1; max_batch <= 300; ++max_batch) {
            const Buckets buckets = decode_graph_buckets(max_batch, steps);
            for (int64_t seqs = 1; seqs <= max_batch; ++seqs) {
                const int64_t padding = bucket_for(seqs, buckets) - seqs;
                EXPECT_GE(padding, 0);
                EXPECT_LT(padding * steps, std::max<int64_t>(seqs, 1))
                    << max_batch << " " << steps << " " << seqs;
            }
        }
    }
}

TEST(GraphBuckets, SeventeenSequencesUnderMaxBatch32) {
    EXPECT_EQ(bucket_for(17, decode_graph_buckets(32, 1)), 32);
    EXPECT_EQ(bucket_for(17, decode_graph_buckets(32, 2)), 24);
    EXPECT_EQ(bucket_for(17, decode_graph_buckets(32, 4)), 20);
    EXPECT_EQ(bucket_for(17, decode_graph_buckets(32, 8)), 18);
}

TEST(GraphBuckets, RejectsMaxBatchBelowOne) {
    EXPECT_THROW(decode_graph_buckets(0, 1), std::invalid_argument);
    EXPECT_THROW(decode_graph_buckets(-1, 2), std::invalid_argument);
    EXPECT_THROW(decode_graph_buckets(-256, 4), std::invalid_argument);
}

TEST(GraphBuckets, RejectsStepsThatAreNotAPowerOfTwo) {
    for (int64_t steps :
         {int64_t(0), int64_t(-1), int64_t(-2), int64_t(3), int64_t(5), int64_t(6), int64_t(12)}) {
        EXPECT_THROW(decode_graph_buckets(32, steps), std::invalid_argument) << steps;
    }
}
