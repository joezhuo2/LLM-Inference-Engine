#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/kernels/attention.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/device.h"

namespace engine::test::pa {

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;

using Kernel = std::function<void(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, engine::Stream)>;

struct Layout {
    int heads, kv_heads, head_dim, block_size;
    int64_t q_width() const { return int64_t(heads) * head_dim; }
    int64_t kv_width() const { return int64_t(kv_heads) * head_dim; }
    int64_t width() const { return q_width() + 2 * kv_width(); }
    int64_t block_elems() const { return kv_width() * block_size; }
};

inline const Layout kTinyLlama{32, 4, 64, 16};
inline constexpr int kMaxContext = 2048;
inline constexpr int64_t kGuard = 256;
inline constexpr double kTolerance = 1.0 / 64;
inline constexpr float kBf16Epsilon = 1.0f / 128.0f;
inline constexpr uint16_t kNan = 0xFFFF;

inline int blocks_for(int len, int block_size) {
    return (len + block_size - 1) / block_size;
}

struct Batch {
    Layout l{};
    int num_blocks = 0;
    int table_width = 0;
    std::vector<int32_t> lens;
    std::vector<int32_t> tables;
    std::vector<uint16_t> qkv, k, v;

    int64_t rows() const { return int64_t(lens.size()); }

    int64_t cell(int64_t row, int64_t pos, int h, int d) const {
        const int64_t block = tables[size_t(row * table_width + pos / l.block_size)];
        return ((block * l.kv_heads + h) * l.block_size + pos % l.block_size) * l.head_dim + d;
    }

    std::vector<bool> reached() const {
        std::vector<bool> used(k.size(), false);
        for (int64_t r = 0; r < rows(); ++r)
            for (int64_t pos = 0; pos < lens[size_t(r)]; ++pos)
                for (int h = 0; h < l.kv_heads; ++h)
                    for (int d = 0; d < l.head_dim; ++d) used[size_t(cell(r, pos, h, d))] = true;
        return used;
    }
};

inline int default_width(const std::vector<int32_t>& lens, int block_size) {
    int widest = 1;
    for (const int32_t len : lens) widest = std::max(widest, blocks_for(len, block_size));
    return std::max(widest, std::min(widest + 2, kMaxContext / block_size));
}

inline void fill_cache(Batch& b, uint32_t seed, float range) {
    const auto cells = size_t(b.num_blocks * b.l.block_elems());
    b.qkv = random_bf16(size_t(b.rows() * b.l.width()), seed + 1, -range, range);
    b.k = random_bf16(cells, seed + 2, -range, range);
    b.v = random_bf16(cells, seed + 3, -range, range);
}

inline Batch make_batch(const Layout& l, std::vector<int32_t> lens, uint32_t seed,
                        int table_width = 0, float range = 2.0f) {
    std::mt19937 rng(seed);
    Batch b;
    b.l = l;
    b.lens = std::move(lens);
    b.table_width = table_width > 0 ? table_width : default_width(b.lens, l.block_size);
    int needed = 0;
    for (const int32_t len : b.lens) needed += blocks_for(len, l.block_size);
    b.num_blocks = needed + 3;
    std::vector<int32_t> ids(static_cast<size_t>(b.num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    b.tables.resize(size_t(b.rows() * b.table_width));
    size_t next = 0;
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int j = 0; j < b.table_width; ++j)
            b.tables[size_t(r * b.table_width + j)] =
                j < blocks_for(b.lens[size_t(r)], l.block_size)
                    ? ids[next++]
                    : int32_t(rng() % uint32_t(b.num_blocks));
    fill_cache(b, seed, range);
    return b;
}

inline Batch make_prefill(const Layout& l, int len, uint32_t seed) {
    std::mt19937 rng(seed);
    Batch b;
    b.l = l;
    b.table_width = default_width({len}, l.block_size);
    b.num_blocks = blocks_for(len, l.block_size) + 3;
    std::vector<int32_t> table(static_cast<size_t>(b.num_blocks));
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    table.resize(size_t(b.table_width), 0);
    for (int t = 0; t < len; ++t) {
        b.lens.push_back(t + 1);
        b.tables.insert(b.tables.end(), table.begin(), table.end());
    }
    fill_cache(b, seed, 2.0f);
    return b;
}

inline Batch single_row(const Batch& b, int64_t row) {
    Batch one = b;
    one.lens = {b.lens[size_t(row)]};
    one.tables.assign(b.tables.begin() + row * b.table_width,
                      b.tables.begin() + (row + 1) * b.table_width);
    one.qkv.assign(b.qkv.begin() + row * b.l.width(), b.qkv.begin() + (row + 1) * b.l.width());
    return one;
}

inline std::vector<uint16_t> run_kernel(const Kernel& kernel, const Batch& b, uint32_t seed) {
    const Layout& l = b.l;
    const int64_t n = b.rows() * l.q_width();
    const auto init = random_bf16(size_t(n + 2 * kGuard), seed);
    DeviceBuffer out_buf = upload(init);
    DeviceBuffer qkv = upload(b.qkv), k = upload(b.k), v = upload(b.v);
    DeviceBuffer tables = upload(b.tables), lens = upload(b.lens);
    const std::initializer_list<int64_t> cache_shape{b.num_blocks, l.kv_heads, l.block_size,
                                                     l.head_dim};

    CudaStream stream;
    kernel(Tensor(static_cast<uint16_t*>(out_buf.data()) + kGuard, DType::BF16,
                  {b.rows(), l.q_width()}),
           Tensor(qkv.data(), DType::BF16, {b.rows(), l.width()}),
           Tensor(k.data(), DType::BF16, cache_shape), Tensor(v.data(), DType::BF16, cache_shape),
           Tensor(tables.data(), DType::I32, {b.rows(), b.table_width}),
           Tensor(lens.data(), DType::I32, {b.rows()}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(download<uint16_t>(qkv), b.qkv) << "qkv was modified";
    EXPECT_EQ(download<uint16_t>(k), b.k) << "k_cache was modified";
    EXPECT_EQ(download<uint16_t>(v), b.v) << "v_cache was modified";
    EXPECT_EQ(download<int32_t>(tables), b.tables) << "block_tables was modified";
    EXPECT_EQ(download<int32_t>(lens), b.lens) << "context_lens was modified";

    const auto after = download<uint16_t>(out_buf);
    for (int64_t i = 0; i < kGuard; ++i) {
        EXPECT_EQ(after[size_t(i)], init[size_t(i)]) << "write before out at " << i - kGuard;
        EXPECT_EQ(after[size_t(kGuard + n + i)], init[size_t(kGuard + n + i)])
            << "write after out at " << i;
    }
    for (int64_t r = 0; r < b.rows(); ++r) {
        if (b.lens[size_t(r)] != 0) continue;
        const auto row = size_t(kGuard + r * l.q_width());
        EXPECT_TRUE(
            std::equal(after.begin() + row, after.begin() + row + l.q_width(), init.begin() + row))
            << "padding row " << r << " was written";
    }
    return {after.begin() + kGuard, after.begin() + kGuard + n};
}

struct Expected {
    std::vector<uint16_t> out;
    std::vector<double> slack;
};

inline Expected reference(const Batch& b) {
    const Layout& l = b.l;
    const int group = l.heads / l.kv_heads;
    const double scale = 1.0 / std::sqrt(double(l.head_dim));
    Expected e{std::vector<uint16_t>(size_t(b.rows() * l.q_width())),
               std::vector<double>(size_t(b.rows() * l.q_width()), -1.0)};
    CudaStream stream;
    for (int64_t r = 0; r < b.rows(); ++r) {
        const int64_t len = b.lens[size_t(r)];
        if (len == 0) continue;
        std::vector<uint16_t> kc(size_t(len * l.kv_width())), vc(kc.size());
        for (int64_t j = 0; j < len; ++j)
            for (int h = 0; h < l.kv_heads; ++h)
                for (int d = 0; d < l.head_dim; ++d) {
                    const auto i = size_t(j * l.kv_width() + h * l.head_dim + d);
                    kc[i] = b.k[size_t(b.cell(r, j, h, d))];
                    vc[i] = b.v[size_t(b.cell(r, j, h, d))];
                }
        const std::vector<uint16_t> q(b.qkv.begin() + r * l.width(),
                                      b.qkv.begin() + (r + 1) * l.width());
        DeviceBuffer dq = upload(q), dk = upload(kc), dv = upload(vc);
        DeviceBuffer dout(size_t(l.q_width()) * sizeof(uint16_t));
        engine::naive::attention(Tensor(dout.data(), DType::BF16, {1, l.q_width()}),
                                 Tensor(dq.data(), DType::BF16, {1, l.width()}),
                                 Tensor(dk.data(), DType::BF16, {len, l.kv_width()}),
                                 Tensor(dv.data(), DType::BF16, {len, l.kv_width()}), len - 1,
                                 l.heads, l.kv_heads, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto naive = download<uint16_t>(dout);

        for (int h = 0; h < l.heads; ++h) {
            const int64_t kv = int64_t(h / group) * l.head_dim;
            std::vector<double> p(static_cast<size_t>(len));
            for (int64_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int d = 0; d < l.head_dim; ++d)
                    dot += double(from_bf16(q[size_t(h * l.head_dim + d)])) *
                           from_bf16(kc[size_t(j * l.kv_width() + kv + d)]);
                p[size_t(j)] = dot * scale;
            }
            const double m = *std::max_element(p.begin(), p.end());
            double sum = 0.0;
            for (double& s : p) sum += s = std::exp(s - m);
            for (int d = 0; d < l.head_dim; ++d) {
                double weighted_abs = 0.0;
                for (int64_t j = 0; j < len; ++j)
                    weighted_abs += p[size_t(j)] / sum *
                                    std::abs(from_bf16(vc[size_t(j * l.kv_width() + kv + d)]));
                const auto local = size_t(h * l.head_dim + d);
                const auto i = size_t(r * l.q_width()) + local;
                e.out[i] = naive[local];
                e.slack[i] = kTolerance * (weighted_abs + std::abs(from_bf16(naive[local])));
            }
        }
    }
    return e;
}

inline void expect_close(const std::vector<uint16_t>& actual, const Expected& e) {
    size_t outside = 0, first = 0;
    double worst = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (e.slack[i] < 0.0) continue;
        const double err = std::abs(double(from_bf16(actual[i])) - from_bf16(e.out[i]));
        if (!(err <= e.slack[i]) && outside++ == 0) first = i;
        if (e.slack[i] > 0.0) worst = std::max(worst, err / e.slack[i]);
    }
    EXPECT_EQ(outside, 0u) << "first at element " << first << ", worst " << worst
                           << " times the allowed error, " << compare_bf16(actual, e.out);
}

inline void check_kernel(const Kernel& kernel, const Batch& b, uint32_t seed) {
    expect_close(run_kernel(kernel, b, seed), reference(b));
}

inline size_t outside_value_scale(const std::vector<uint16_t>& actual,
                                  const std::vector<uint16_t>& expected,
                                  const std::vector<uint16_t>& v, int heads, int kv_heads,
                                  int head_dim) {
    const int64_t q_width = int64_t(heads) * head_dim, kv_width = int64_t(kv_heads) * head_dim;
    const int64_t tokens = int64_t(actual.size()) / q_width;
    const int group = heads / kv_heads;
    std::vector<float> v_max(size_t(kv_width), 0.0f);
    size_t n = 0;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t c = 0; c < kv_width; ++c)
            v_max[size_t(c)] =
                std::max(v_max[size_t(c)], std::abs(from_bf16(v[size_t(t * kv_width + c)])));
        for (int64_t c = 0; c < q_width; ++c) {
            const auto i = size_t(t * q_width + c);
            const int64_t kv_c = (c / head_dim) / group * head_dim + c % head_dim;
            if (std::abs(from_bf16(actual[i]) - from_bf16(expected[i])) >
                kBf16Epsilon * v_max[size_t(kv_c)])
                ++n;
        }
    }
    return n;
}

}  // namespace engine::test::pa
