#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/paged_runner.h"
#include "engine/sched/paged_batch.h"
#include "support/device.h"
#include "support/logits.h"
#include "support/model.h"

using engine::DecodeGraphs;
using engine::KvCache;
using engine::PagedBatch;
using engine::PagedRunner;
using engine::Request;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::Tensor;
using engine::test::argmax_rows;
using engine::test::CudaStream;
using engine::test::relative_l2;

namespace {

constexpr int kBlockSize = 16;
constexpr int kMaxSeqs = 16;
const std::vector<int64_t> kBuckets = {1, 2, 4, 8, 16};

std::vector<uint16_t> to_host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<int32_t> prompt_of(int64_t id, int length, int vocab) {
    std::vector<int32_t> p;
    for (int i = 0; i < length; ++i)
        p.push_back(int32_t(1 + (id * 7919 + i * 104729) % (vocab - 1)));
    return p;
}

struct Rig {
    Rig(const engine::DeviceWeights& w, cudaStream_t stream, int max_context)
        : cache(w.config, 400, kBlockSize),
          runner(w.config, w.weights, cache, 256, kMaxSeqs, max_context, stream),
          scheduler(SchedulerConfig{.num_blocks = 400,
                                    .block_size = kBlockSize,
                                    .max_batch = kMaxSeqs,
                                    .max_prefill_tokens = 256,
                                    .max_context = max_context}),
          vocab(w.config.vocab) {}

    void submit(int64_t id, int prompt_len, int max_new_tokens) {
        Request r{.id = id, .max_new_tokens = max_new_tokens};
        r.prompt = prompt_of(id, prompt_len, int(vocab));
        scheduler.submit(r);
    }

    PagedBatch next_decode_batch() {
        for (;;) {
            const auto step = scheduler.schedule();
            const auto batch = engine::build_paged_batch(step, scheduler, int(runner.max_blocks()));
            if (!step.prefill) return batch;
            scheduler.update(argmax_rows(to_host(runner.forward(batch)), vocab));
        }
    }

    std::vector<uint16_t> padded_eager(const PagedBatch& batch, int64_t bucket) {
        runner.stage(engine::pad_decode_batch(batch, bucket));
        const Tensor logits = runner.run_staged(bucket, bucket);
        std::vector<uint16_t> all = to_host(logits);
        all.resize(size_t(batch.num_seqs * vocab));
        return all;
    }

    KvCache cache;
    PagedRunner runner;
    Scheduler scheduler;
    int64_t vocab;
};

int64_t bucket_of(int64_t seqs) {
    for (const int64_t b : kBuckets)
        if (b >= seqs) return b;
    return 0;
}

struct Arrival {
    int64_t id;
    int prompt_len, max_new_tokens;
    int at_iteration;
};

std::map<int64_t, std::vector<int32_t>> run_to_completion(Rig& rig, DecodeGraphs* graphs,
                                                          const std::vector<Arrival>& arrivals) {
    std::map<int64_t, std::vector<int32_t>> outputs;
    size_t submitted = 0;
    for (int iteration = 0; submitted < arrivals.size() ||
                            rig.scheduler.num_waiting() + rig.scheduler.num_running() > 0;
         ++iteration) {
        for (; submitted < arrivals.size() && arrivals[submitted].at_iteration <= iteration;
             ++submitted)
            rig.submit(arrivals[submitted].id, arrivals[submitted].prompt_len,
                       arrivals[submitted].max_new_tokens);
        if (rig.scheduler.num_waiting() + rig.scheduler.num_running() == 0) continue;

        const auto step = rig.scheduler.schedule();
        const auto batch =
            engine::build_paged_batch(step, rig.scheduler, int(rig.runner.max_blocks()));
        std::vector<uint16_t> logits;
        if (step.prefill) {
            logits = to_host(rig.runner.forward(batch));
        } else if (graphs != nullptr) {
            logits = to_host(graphs->forward(batch));
        } else {
            logits = rig.padded_eager(batch, bucket_of(batch.num_seqs));
        }
        rig.scheduler.update(argmax_rows(logits, rig.vocab));
        for (auto& r : rig.scheduler.take_finished()) outputs[r.id] = r.output;
    }
    return outputs;
}

std::vector<std::vector<uint16_t>> snapshot(KvCache& cache) {
    std::vector<std::vector<uint16_t>> planes;
    for (int layer = 0; layer < cache.layers(); ++layer) {
        planes.push_back(to_host(cache.k(layer)));
        planes.push_back(to_host(cache.v(layer)));
    }
    return planes;
}

class DecodeGraphsTest : public ::testing::Test {
protected:
    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }
    CudaStream stream;
};

}  // namespace

TEST_F(DecodeGraphsTest, ReplayIsBitIdenticalToThePaddedEagerStep) {
    for (const int n : {1, 2, 3, 5, 8, 13, 16}) {
        SCOPED_TRACE("batch of " + std::to_string(n));
        Rig rig(w(), stream, 128);
        for (int i = 0; i < n; ++i) rig.submit(i, 4 + (i * 3) % 17, 8);
        const PagedBatch batch = rig.next_decode_batch();
        ASSERT_EQ(batch.num_seqs, n);

        DecodeGraphs graphs(rig.runner, kBuckets);
        EXPECT_EQ(graphs.num_captured(), int64_t(kBuckets.size()));
        ASSERT_TRUE(graphs.covers(batch));
        const Tensor logits = graphs.forward(batch);
        ASSERT_EQ(logits.ndim, 2);
        EXPECT_EQ(logits.shape[0], n);
        EXPECT_EQ(logits.shape[1], rig.vocab);
        const auto replayed = to_host(logits);
        EXPECT_EQ(replayed, rig.padded_eager(batch, bucket_of(n)));
    }
}

TEST_F(DecodeGraphsTest, ReplayIsCloseToTheUnpaddedEagerStep) {
    Rig rig(w(), stream, 128);
    for (int i = 0; i < 13; ++i) rig.submit(i, 5 + (i * 7) % 23, 8);
    const PagedBatch batch = rig.next_decode_batch();
    DecodeGraphs graphs(rig.runner, kBuckets);
    const auto replayed = to_host(graphs.forward(batch));
    const auto eager = to_host(rig.runner.forward(batch));
    EXPECT_LE(relative_l2(replayed, eager), 0.05);
    EXPECT_EQ(argmax_rows(replayed, rig.vocab).size(), 13u);
}

TEST_F(DecodeGraphsTest, ACompleteGreedyDecodeMatchesThePaddedEagerLoop) {
    std::vector<Arrival> arrivals;
    for (int i = 0; i < 16; ++i)
        arrivals.push_back({i, 3 + (i * 5) % 19, 3 + (i * 7) % 24, 4 * (i / 4)});
    Rig eager_rig(w(), stream, 128);
    const auto want = run_to_completion(eager_rig, nullptr, arrivals);
    ASSERT_EQ(want.size(), 16u);

    Rig graph_rig(w(), stream, 128);
    DecodeGraphs graphs(graph_rig.runner, kBuckets);
    const auto got = run_to_completion(graph_rig, &graphs, arrivals);
    EXPECT_EQ(got, want);
}

TEST_F(DecodeGraphsTest, ConstructionLeavesTheKvCacheExactlyAsItWas) {
    Rig rig(w(), stream, 128);
    for (int i = 0; i < 3; ++i) rig.submit(i, 9 + i, 8);
    const PagedBatch batch = rig.next_decode_batch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto before = snapshot(rig.cache);
    {
        DecodeGraphs graphs(rig.runner, kBuckets);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(snapshot(rig.cache), before);
    }
    EXPECT_EQ(snapshot(rig.cache), before);
    (void)batch;
}

TEST_F(DecodeGraphsTest, TwoHundredReplaysWithChangingInputsStayBitIdentical) {
    Rig rig(w(), stream, 256);
    rig.submit(1, 3, 215);
    rig.submit(2, 40, 215);
    DecodeGraphs graphs(rig.runner, kBuckets);
    int replays = 0;
    while (replays < 200) {
        const PagedBatch batch = rig.next_decode_batch();
        const auto replayed = to_host(graphs.forward(batch));
        ASSERT_EQ(replayed, rig.padded_eager(batch, bucket_of(batch.num_seqs)))
            << "replay " << replays;
        rig.scheduler.update(argmax_rows(replayed, rig.vocab));
        ++replays;
    }
}

TEST_F(DecodeGraphsTest, CoversAndBucketFor) {
    Rig rig(w(), stream, 128);
    for (int i = 0; i < 5; ++i) rig.submit(i, 6, 4);
    const PagedBatch five = rig.next_decode_batch();
    DecodeGraphs graphs(rig.runner, {1, 2, 4});
    EXPECT_EQ(graphs.num_captured(), 3);
    EXPECT_EQ(graphs.bucket_for(1), 1);
    EXPECT_EQ(graphs.bucket_for(2), 2);
    EXPECT_EQ(graphs.bucket_for(3), 4);
    EXPECT_EQ(graphs.bucket_for(4), 4);
    EXPECT_EQ(graphs.bucket_for(5), 0);

    EXPECT_FALSE(graphs.covers(five));
    EXPECT_THROW(graphs.forward(five), std::invalid_argument);

    PagedBatch prefill = five;
    prefill.prefill = true;
    EXPECT_FALSE(graphs.covers(prefill));
    EXPECT_THROW(graphs.forward(prefill), std::invalid_argument);

    PagedBatch four = five;
    four.num_seqs = 4;
    four.token_ids.resize(4);
    four.positions.resize(4);
    four.slot_mapping.resize(4);
    four.context_lens.resize(4);
    four.logit_rows.resize(4);
    four.block_tables.resize(size_t(4 * four.max_blocks));
    EXPECT_TRUE(graphs.covers(four));
    PagedBatch narrow = four;
    narrow.max_blocks = four.max_blocks - 1;
    EXPECT_FALSE(graphs.covers(narrow));
    EXPECT_THROW(graphs.forward(narrow), std::invalid_argument);
    EXPECT_NO_THROW(graphs.forward(four));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST_F(DecodeGraphsTest, SingleBucketListsWork) {
    Rig rig(w(), stream, 128);
    for (int i = 0; i < 3; ++i) rig.submit(i, 7 + i, 4);
    const PagedBatch batch = rig.next_decode_batch();
    {
        DecodeGraphs one(rig.runner, {16});
        EXPECT_EQ(one.bucket_for(3), 16);
        EXPECT_EQ(to_host(one.forward(batch)), rig.padded_eager(batch, 16));
    }
    {
        DecodeGraphs one(rig.runner, {1});
        EXPECT_EQ(one.num_captured(), 1);
        EXPECT_FALSE(one.covers(batch));
    }
}

TEST_F(DecodeGraphsTest, RejectsBadBucketLists) {
    Rig rig(w(), stream, 128);
    for (const std::vector<int64_t>& buckets :
         {std::vector<int64_t>{}, std::vector<int64_t>{0}, std::vector<int64_t>{-1, 2},
          std::vector<int64_t>{2, 2}, std::vector<int64_t>{4, 2}, std::vector<int64_t>{1, 17},
          std::vector<int64_t>{32}}) {
        EXPECT_THROW(DecodeGraphs(rig.runner, buckets), std::invalid_argument);
    }
}
