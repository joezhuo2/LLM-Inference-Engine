#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "engine/kernels/write_kv.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/device.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::write_kv;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::random_bf16;
using engine::test::upload;

namespace {

struct Case {
    int heads, kv_heads, head_dim, num_blocks, block_size;
    int64_t kv_width() const { return int64_t(kv_heads) * head_dim; }
    int64_t width() const { return int64_t(heads) * head_dim + 2 * kv_width(); }
    int64_t cache_elems() const { return int64_t(num_blocks) * block_size * kv_width(); }
};

constexpr int64_t kGuard = 256;

std::vector<int32_t> sequence_slots(const std::vector<int>& table, int length, int block_size) {
    std::vector<int32_t> slots;
    for (int pos = 0; pos < length; ++pos)
        slots.push_back(table[size_t(pos / block_size)] * block_size + pos % block_size);
    return slots;
}

std::vector<int> random_blocks(int count, int num_blocks, std::mt19937& rng) {
    std::vector<int> ids(static_cast<size_t>(num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(size_t(count));
    return ids;
}

void check(const Case& c, const std::vector<int32_t>& slots, uint32_t seed) {
    const auto tokens = int64_t(slots.size());
    const int64_t cache = c.cache_elems();
    const int64_t k_begin = kGuard, v_begin = 2 * kGuard + cache;
    const int64_t total = 3 * kGuard + 2 * cache;

    const auto qkv_host = random_bf16(size_t(std::max<int64_t>(tokens, 1) * c.width()), seed);
    const auto memory_host = random_bf16(size_t(total), seed + 1, -8.0f, 8.0f);
    DeviceBuffer qkv_buf = upload(qkv_host);
    DeviceBuffer memory = upload(memory_host);
    DeviceBuffer slot_buf = upload(slots.empty() ? std::vector<int32_t>{7} : slots);

    auto* base = static_cast<uint16_t*>(memory.data());
    const Tensor k(base + k_begin, DType::BF16,
                   {c.num_blocks, c.kv_heads, c.block_size, c.head_dim});
    const Tensor v(base + v_begin, DType::BF16,
                   {c.num_blocks, c.kv_heads, c.block_size, c.head_dim});
    const Tensor qkv(qkv_buf.data(), DType::BF16, {tokens, c.width()});
    const Tensor slot_mapping(slot_buf.data(), DType::I32, {tokens});

    CudaStream stream;
    write_kv(k, v, qkv, slot_mapping, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint16_t> expected = memory_host;
    for (int64_t t = 0; t < tokens; ++t) {
        const int32_t s = slots[size_t(t)];
        if (s < 0) continue;
        const int64_t block = s / c.block_size, offset = s % c.block_size;
        for (int h = 0; h < c.kv_heads; ++h)
            for (int d = 0; d < c.head_dim; ++d) {
                const int64_t cell =
                    ((block * c.kv_heads + h) * c.block_size + offset) * c.head_dim + d;
                const int64_t col = h * int64_t(c.head_dim) + d;
                const int64_t row = t * c.width() + c.width() - 2 * c.kv_width();
                expected[size_t(k_begin + cell)] = qkv_host[size_t(row + col)];
                expected[size_t(v_begin + cell)] = qkv_host[size_t(row + c.kv_width() + col)];
            }
    }

    const auto actual = download<uint16_t>(memory);
    int64_t wrong = 0, first = -1;
    for (int64_t i = 0; i < total; ++i)
        if (actual[size_t(i)] != expected[size_t(i)] && wrong++ == 0) first = i;
    EXPECT_EQ(wrong, 0) << "first wrong element " << first << " (K starts at " << k_begin
                        << ", V at " << v_begin << ", each " << cache << " elements)";
    EXPECT_EQ(download<uint16_t>(qkv_buf), qkv_host) << "qkv was modified";
    if (!slots.empty()) {
        EXPECT_EQ(download<int32_t>(slot_buf), slots) << "slot_mapping was modified";
    }
}

}  // namespace

TEST(WriteKv, SequenceThroughAShuffledBlockTable) {
    const Case c{32, 4, 64, 9, 16};
    check(c, sequence_slots({5, 2, 8}, 37, 16), 1);
}

TEST(WriteKv, PaddingRowsWriteNothing) {
    const Case c{32, 4, 64, 9, 16};
    auto slots = sequence_slots({3, 7, 1}, 40, 16);
    for (size_t i = 0; i < slots.size(); i += 3) slots[i] = -1;
    slots.insert(slots.end(), {-1, -1, -1});
    check(c, slots, 2);
    check(c, std::vector<int32_t>(5, -1), 3);
}

TEST(WriteKv, BoundarySlots) {
    const Case c{32, 4, 64, 6, 16};
    check(c, {0, 15, 16, 31, 95, 80, 48, 47}, 4);
}

TEST(WriteKv, OddBlockSizesAndHeadLayouts) {
    check({6, 3, 6, 7, 1}, {6, 0, 3, 5}, 5);
    std::mt19937 rng(6);
    std::vector<int32_t> slots(35);
    std::iota(slots.begin(), slots.end(), 0);
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(20);
    check({6, 3, 6, 7, 5}, slots, 7);
    check({1, 1, 1, 3, 3}, {8, 4, 0}, 8);
}

TEST(WriteKv, FullLengthPrefill) {
    std::mt19937 rng(9);
    check({32, 4, 64, 200, 16}, sequence_slots(random_blocks(128, 200, rng), 2048, 16), 10);
}

TEST(WriteKv, DecodeBatchOfSixteenSequences) {
    std::mt19937 rng(11);
    const Case c{32, 4, 64, 100, 16};
    const auto blocks = random_blocks(96, 100, rng);
    std::vector<int32_t> slots;
    for (int seq = 0; seq < 16; ++seq) {
        const std::vector<int> table(blocks.begin() + seq * 6, blocks.begin() + seq * 6 + 6);
        const int length = 1 + int(rng() % 96);
        slots.push_back(sequence_slots(table, length, 16).back());
    }
    check(c, slots, 12);
}

TEST(WriteKv, NoTokensIsANoOp) {
    check({32, 4, 64, 4, 16}, {}, 13);
}

TEST(WriteKv, RejectsBadArguments) {
    const Case c{4, 2, 8, 3, 4};
    DeviceBuffer memory(size_t(2 * c.cache_elems()) * 2);
    DeviceBuffer qkv_buf(size_t(5 * c.width()) * 2);
    DeviceBuffer slot_buf(5 * sizeof(int32_t));
    auto* base = static_cast<uint16_t*>(memory.data());
    const Tensor k(base, DType::BF16, {3, 2, 4, 8});
    const Tensor v(base + c.cache_elems(), DType::BF16, {3, 2, 4, 8});
    const Tensor qkv(qkv_buf.data(), DType::BF16, {5, c.width()});
    const Tensor slots(slot_buf.data(), DType::I32, {5});
    const engine::Stream s = nullptr;

    EXPECT_THROW(write_kv(Tensor(base, DType::F16, {3, 2, 4, 8}), v, qkv, slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, Tensor(base, DType::F32, {3, 2, 4, 8}), qkv, slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(Tensor(base, DType::BF16, {3, 2, 32}), v, qkv, slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, Tensor(base, DType::BF16, {3, 2, 8, 4}), qkv, slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, Tensor(base, DType::BF16, {2, 2, 4, 8}), qkv, slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, v, Tensor(qkv_buf.data(), DType::F32, {5, c.width()}), slots, s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, v, Tensor(qkv_buf.data(), DType::BF16, {5 * c.width()}), slots, s),
                 std::invalid_argument);
    EXPECT_THROW(
        write_kv(k, v, Tensor(qkv_buf.data(), DType::BF16, {5, 2 * c.kv_width()}), slots, s),
        std::invalid_argument);
    EXPECT_THROW(write_kv(k, v, qkv, Tensor(slot_buf.data(), DType::I64, {5}), s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, v, qkv, Tensor(slot_buf.data(), DType::I32, {5, 1}), s),
                 std::invalid_argument);
    EXPECT_THROW(write_kv(k, v, qkv, Tensor(slot_buf.data(), DType::I32, {4}), s),
                 std::invalid_argument);
}
