#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loader/tensor_parallel.h"
#include "engine/loader/weight_layout.h"
#include "support/fake_checkpoint.h"

using engine::CopyOp;
using engine::kWeightAlignment;
using engine::LayerLayout;
using engine::ModelConfig;
using engine::plan_shard_layout;
using engine::plan_weight_layout;
using engine::Region;
using engine::SafetensorsHeader;
using engine::WeightLayout;
using engine::test::fake_header;
using engine::test::tinyllama_config;

namespace {

using Shape = std::vector<int64_t>;

ModelConfig tp_config() {
    ModelConfig c = engine::test::tiny_config();
    c.hidden = 32;
    c.heads = 8;
    c.kv_heads = 4;
    c.head_dim = 4;
    c.intermediate = 24;
    return c;
}

std::vector<Region> regions(const WeightLayout& w) {
    std::vector<Region> r = {w.embed};
    for (const auto& l : w.layers) r.insert(r.end(), {l.ln1, l.wqkv, l.wo, l.ln2, l.wgu, l.wdown});
    r.push_back(w.final_norm);
    r.push_back(w.lm_head);
    return r;
}

std::vector<uint16_t> fake_file(const SafetensorsHeader& h) {
    uint64_t end = 0;
    for (const auto& [name, t] : h.tensors) end = std::max(end, t.offset + t.bytes);
    std::vector<uint16_t> file(end / 2);
    std::mt19937 rng(202);
    for (uint16_t& word : file) word = uint16_t(rng());
    return file;
}

struct Loaded {
    WeightLayout layout;
    std::vector<uint16_t> words;
    std::vector<uint8_t> writes;

    uint16_t at(const Region& region, int64_t row, int64_t col) const {
        const int64_t cols = region.shape.size() == 2 ? region.shape[1] : 1;
        return words.at(region.offset / 2 + uint64_t(row * cols + col));
    }
};

Loaded load(const WeightLayout& layout, const std::vector<uint16_t>& file) {
    Loaded loaded{layout, std::vector<uint16_t>(layout.total_bytes / 2, 0),
                  std::vector<uint8_t>(layout.total_bytes, 0)};
    for (const CopyOp& c : layout.copies) {
        EXPECT_LE(c.src + c.bytes, file.size() * 2) << c.name;
        EXPECT_LE(c.dst + c.bytes, layout.total_bytes) << c.name;
        if (c.src + c.bytes > file.size() * 2 || c.dst + c.bytes > layout.total_bytes) continue;
        std::memcpy(reinterpret_cast<char*>(loaded.words.data()) + c.dst,
                    reinterpret_cast<const char*>(file.data()) + c.src, c.bytes);
        for (uint64_t i = 0; i < c.bytes; ++i) loaded.writes[c.dst + i]++;
    }
    return loaded;
}

struct Fixture {
    ModelConfig config = tp_config();
    SafetensorsHeader header = fake_header(config, 64);
    std::vector<uint16_t> file = fake_file(header);
    Loaded full = load(plan_weight_layout(config, header), file);

    Loaded shard(int rank, int num_ranks) const {
        return load(plan_shard_layout(config, header, rank, num_ranks), file);
    }
};

void expect_same_region(const Loaded& a, const Region& ra, const Loaded& b, const Region& rb) {
    ASSERT_EQ(ra.shape, rb.shape);
    ASSERT_EQ(ra.bytes, rb.bytes);
    for (uint64_t i = 0; i < ra.bytes / 2; ++i) {
        ASSERT_EQ(a.words.at(ra.offset / 2 + i), b.words.at(rb.offset / 2 + i)) << "word " << i;
    }
}

void expect_rejected(const ModelConfig& config, const SafetensorsHeader& header, int rank,
                     int num_ranks) {
    try {
        plan_shard_layout(config, header, rank, num_ranks);
        ADD_FAILURE() << "rank " << rank << " of " << num_ranks << " was accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("plan_shard_layout"), std::string::npos) << e.what();
    }
}

void expect_runtime_error(const ModelConfig& config, const SafetensorsHeader& header,
                          const std::string& fragment) {
    try {
        plan_shard_layout(config, header, 0, 2);
        ADD_FAILURE() << "expected a rejection mentioning '" << fragment << "'";
    } catch (const std::invalid_argument& e) {
        ADD_FAILURE() << "expected std::runtime_error, got invalid_argument: " << e.what();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

double value(uint16_t word) {
    return double(word % 7) - 3.0;
}

TEST(TensorParallel, OneRankIsTheFullLayout) {
    const Fixture f;
    const Loaded one = f.shard(0, 1);
    const auto a = regions(one.layout), b = regions(f.full.layout);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(a[i].offset, b[i].offset);
        expect_same_region(one, a[i], f.full, b[i]);
    }
    EXPECT_EQ(one.layout.total_bytes, f.full.layout.total_bytes);
}

TEST(TensorParallel, RegionShapes) {
    const Fixture f;
    const ModelConfig& c = f.config;
    for (const int n : {1, 2, 4}) {
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const WeightLayout w = plan_shard_layout(c, f.header, rank, n);
            const int64_t h = c.heads / n, k = c.kv_heads / n, ff = c.intermediate / n;
            EXPECT_EQ(w.embed.shape, (Shape{c.vocab, c.hidden}));
            EXPECT_EQ(w.final_norm.shape, (Shape{c.hidden}));
            EXPECT_EQ(w.lm_head.shape, (Shape{c.vocab, c.hidden}));
            ASSERT_EQ(int(w.layers.size()), c.layers);
            for (const LayerLayout& l : w.layers) {
                EXPECT_EQ(l.ln1.shape, (Shape{c.hidden}));
                EXPECT_EQ(l.wqkv.shape, (Shape{(h + 2 * k) * c.head_dim, c.hidden}));
                EXPECT_EQ(l.wo.shape, (Shape{c.hidden, h * c.head_dim}));
                EXPECT_EQ(l.ln2.shape, (Shape{c.hidden}));
                EXPECT_EQ(l.wgu.shape, (Shape{2 * ff, c.hidden}));
                EXPECT_EQ(l.wdown.shape, (Shape{c.hidden, ff}));
            }
        }
    }
}

TEST(TensorParallel, RegionsAreAlignedOrderedAndDisjoint) {
    const Fixture f;
    for (const int n : {2, 4}) {
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const WeightLayout w = plan_shard_layout(f.config, f.header, rank, n);
            const auto r = regions(w);
            for (size_t i = 0; i < r.size(); ++i) {
                uint64_t bytes = 2;
                for (const int64_t dim : r[i].shape) bytes *= uint64_t(dim);
                EXPECT_EQ(r[i].bytes, bytes) << "region " << i;
                EXPECT_EQ(r[i].offset % kWeightAlignment, 0u) << "region " << i;
                if (i + 1 < r.size()) {
                    EXPECT_LE(r[i].offset + r[i].bytes, r[i + 1].offset) << "region " << i;
                    EXPECT_LT(r[i + 1].offset, r[i].offset + r[i].bytes + kWeightAlignment)
                        << "region " << i;
                }
            }
            EXPECT_EQ(r.front().offset, 0u);
            EXPECT_EQ(w.total_bytes, r.back().offset + r.back().bytes);
            EXPECT_LT(w.total_bytes, f.full.layout.total_bytes);
        }
    }
}

TEST(TensorParallel, EveryRegionByteIsWrittenExactlyOnce) {
    const Fixture f;
    for (const int n : {1, 2, 4}) {
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const Loaded s = f.shard(rank, n);
            std::vector<uint8_t> expected(s.layout.total_bytes, 0);
            for (const Region& region : regions(s.layout)) {
                std::fill_n(expected.begin() + int64_t(region.offset), region.bytes, uint8_t(1));
            }
            ASSERT_EQ(s.writes, expected);
        }
    }
}

TEST(TensorParallel, CopiesReadTheTensorTheyNameAndAreSorted) {
    const Fixture f;
    for (int rank = 0; rank < 4; ++rank) {
        SCOPED_TRACE(rank);
        const WeightLayout w = plan_shard_layout(f.config, f.header, rank, 4);
        ASSERT_FALSE(w.copies.empty());
        for (size_t i = 0; i < w.copies.size(); ++i) {
            const CopyOp& c = w.copies[i];
            ASSERT_TRUE(f.header.tensors.contains(c.name)) << c.name;
            const auto& t = f.header.tensors.at(c.name);
            EXPECT_GE(c.src, t.offset) << c.name;
            EXPECT_LE(c.src + c.bytes, t.offset + t.bytes) << c.name;
            uint64_t bytes = 2;
            for (const int64_t dim : c.shape) bytes *= uint64_t(dim);
            EXPECT_EQ(c.bytes, bytes) << c.name;
            EXPECT_GT(c.bytes, 0u) << c.name;
            if (i > 0) {
                EXPECT_LE(w.copies[i - 1].src, c.src) << c.name;
            }
        }
    }
}

TEST(TensorParallel, QkvHoldsTheRowsOfTheRanksHeads) {
    const Fixture f;
    const ModelConfig& c = f.config;
    for (const int n : {2, 4}) {
        const int64_t q_rows = c.heads / n * c.head_dim, kv_rows = c.kv_heads / n * c.head_dim;
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const Loaded s = f.shard(rank, n);
            for (int layer = 0; layer < c.layers; ++layer) {
                const Region& mine = s.layout.layers[layer].wqkv;
                const Region& whole = f.full.layout.layers[layer].wqkv;
                for (int64_t col = 0; col < c.hidden; ++col) {
                    for (int64_t row = 0; row < q_rows; ++row) {
                        ASSERT_EQ(s.at(mine, row, col), f.full.at(whole, rank * q_rows + row, col));
                    }
                    for (int64_t row = 0; row < kv_rows; ++row) {
                        ASSERT_EQ(s.at(mine, q_rows + row, col),
                                  f.full.at(whole, c.hidden + rank * kv_rows + row, col));
                        ASSERT_EQ(
                            s.at(mine, q_rows + kv_rows + row, col),
                            f.full.at(whole, c.hidden + c.kv_dim() + rank * kv_rows + row, col));
                    }
                }
            }
        }
    }
}

TEST(TensorParallel, OutputProjectionHoldsTheColumnsOfTheRanksHeads) {
    const Fixture f;
    const ModelConfig& c = f.config;
    for (const int n : {2, 4}) {
        const int64_t cols = c.heads / n * c.head_dim;
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const Loaded s = f.shard(rank, n);
            for (int layer = 0; layer < c.layers; ++layer) {
                for (int64_t row = 0; row < c.hidden; ++row) {
                    for (int64_t col = 0; col < cols; ++col) {
                        ASSERT_EQ(s.at(s.layout.layers[layer].wo, row, col),
                                  f.full.at(f.full.layout.layers[layer].wo, row, rank * cols + col))
                            << "row " << row << " col " << col;
                    }
                }
            }
        }
    }
}

TEST(TensorParallel, GateAndUpHoldTheRowsOfTheRanksUnits) {
    const Fixture f;
    const ModelConfig& c = f.config;
    for (const int n : {2, 4}) {
        const int64_t units = c.intermediate / n;
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const Loaded s = f.shard(rank, n);
            for (int layer = 0; layer < c.layers; ++layer) {
                const Region& mine = s.layout.layers[layer].wgu;
                const Region& whole = f.full.layout.layers[layer].wgu;
                for (int64_t row = 0; row < units; ++row) {
                    for (int64_t col = 0; col < c.hidden; ++col) {
                        ASSERT_EQ(s.at(mine, row, col), f.full.at(whole, rank * units + row, col));
                        ASSERT_EQ(s.at(mine, units + row, col),
                                  f.full.at(whole, c.intermediate + rank * units + row, col));
                    }
                }
            }
        }
    }
}

TEST(TensorParallel, DownHoldsTheColumnsOfTheRanksUnits) {
    const Fixture f;
    const ModelConfig& c = f.config;
    for (const int n : {2, 4}) {
        const int64_t units = c.intermediate / n;
        for (int rank = 0; rank < n; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank " << rank << " of " << n);
            const Loaded s = f.shard(rank, n);
            for (int layer = 0; layer < c.layers; ++layer) {
                for (int64_t row = 0; row < c.hidden; ++row) {
                    for (int64_t col = 0; col < units; ++col) {
                        ASSERT_EQ(
                            s.at(s.layout.layers[layer].wdown, row, col),
                            f.full.at(f.full.layout.layers[layer].wdown, row, rank * units + col))
                            << "row " << row << " col " << col;
                    }
                }
            }
        }
    }
}

TEST(TensorParallel, EveryRankHoldsTheUnsplitTensorsWhole) {
    const Fixture f;
    for (int rank = 0; rank < 4; ++rank) {
        SCOPED_TRACE(rank);
        const Loaded s = f.shard(rank, 4);
        expect_same_region(s, s.layout.embed, f.full, f.full.layout.embed);
        expect_same_region(s, s.layout.final_norm, f.full, f.full.layout.final_norm);
        expect_same_region(s, s.layout.lm_head, f.full, f.full.layout.lm_head);
        for (int layer = 0; layer < f.config.layers; ++layer) {
            expect_same_region(s, s.layout.layers[layer].ln1, f.full,
                               f.full.layout.layers[layer].ln1);
            expect_same_region(s, s.layout.layers[layer].ln2, f.full,
                               f.full.layout.layers[layer].ln2);
        }
    }
}

TEST(TensorParallel, PartialOutputsAddUpToTheFullLayer) {
    const Fixture f;
    const ModelConfig& c = f.config;
    std::mt19937 rng(7);
    std::vector<double> x(c.hidden), attn(c.hidden);
    for (double& v : x) v = double(int(rng() % 9)) - 4.0;
    for (double& v : attn) v = double(int(rng() % 9)) - 4.0;
    const auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };

    for (const int n : {2, 4}) {
        SCOPED_TRACE(n);
        const int64_t cols = c.heads / n * c.head_dim, units = c.intermediate / n;
        std::vector<Loaded> shards;
        for (int rank = 0; rank < n; ++rank) shards.push_back(f.shard(rank, n));
        for (int layer = 0; layer < c.layers; ++layer) {
            const LayerLayout& whole = f.full.layout.layers[layer];
            for (int64_t out = 0; out < c.hidden; ++out) {
                double full_o = 0.0, full_mlp = 0.0;
                for (int64_t j = 0; j < c.hidden; ++j) {
                    full_o += attn[j] * value(f.full.at(whole.wo, out, j));
                }
                for (int64_t u = 0; u < c.intermediate; ++u) {
                    double gate = 0.0, up = 0.0;
                    for (int64_t j = 0; j < c.hidden; ++j) {
                        gate += x[j] * value(f.full.at(whole.wgu, u, j));
                        up += x[j] * value(f.full.at(whole.wgu, c.intermediate + u, j));
                    }
                    full_mlp += silu(gate) * up * value(f.full.at(whole.wdown, out, u));
                }
                double sum_o = 0.0, sum_mlp = 0.0;
                for (int rank = 0; rank < n; ++rank) {
                    const Loaded& s = shards[rank];
                    const LayerLayout& mine = s.layout.layers[layer];
                    for (int64_t j = 0; j < cols; ++j) {
                        sum_o += attn[rank * cols + j] * value(s.at(mine.wo, out, j));
                    }
                    for (int64_t u = 0; u < units; ++u) {
                        double gate = 0.0, up = 0.0;
                        for (int64_t j = 0; j < c.hidden; ++j) {
                            gate += x[j] * value(s.at(mine.wgu, u, j));
                            up += x[j] * value(s.at(mine.wgu, units + u, j));
                        }
                        sum_mlp += silu(gate) * up * value(s.at(mine.wdown, out, u));
                    }
                }
                ASSERT_DOUBLE_EQ(sum_o, full_o) << "layer " << layer << " out " << out;
                ASSERT_NEAR(sum_mlp, full_mlp, 1e-9 * (1.0 + std::abs(full_mlp)))
                    << "layer " << layer << " out " << out;
            }
        }
    }
}

TEST(TensorParallel, TinyLlamaShapes) {
    const ModelConfig c = tinyllama_config();
    const SafetensorsHeader h = fake_header(c);
    const WeightLayout full = plan_weight_layout(c, h);
    for (const int n : {2, 4}) {
        SCOPED_TRACE(n);
        const WeightLayout w = plan_shard_layout(c, h, n - 1, n);
        ASSERT_EQ(w.layers.size(), 22u);
        EXPECT_EQ(w.layers[21].wqkv.shape, (Shape{(32 / n + 2 * (4 / n)) * 64, 2048}));
        EXPECT_EQ(w.layers[21].wo.shape, (Shape{2048, 2048 / n}));
        EXPECT_EQ(w.layers[21].wgu.shape, (Shape{2 * 5632 / n, 2048}));
        EXPECT_EQ(w.layers[21].wdown.shape, (Shape{2048, 5632 / n}));
        uint64_t layer_bytes = 0, full_layer_bytes = 0, copied = 0, region_bytes = 0;
        for (const LayerLayout& l : w.layers) {
            layer_bytes += l.wqkv.bytes + l.wo.bytes + l.wgu.bytes + l.wdown.bytes;
        }
        for (const LayerLayout& l : full.layers) {
            full_layer_bytes += l.wqkv.bytes + l.wo.bytes + l.wgu.bytes + l.wdown.bytes;
        }
        EXPECT_EQ(layer_bytes * uint64_t(n), full_layer_bytes);
        for (const CopyOp& copy : w.copies) copied += copy.bytes;
        for (const Region& region : regions(w)) region_bytes += region.bytes;
        EXPECT_EQ(copied, region_bytes);
    }
}

TEST(TensorParallel, RejectsBadRanks) {
    const Fixture f;
    expect_rejected(f.config, f.header, 0, 0);
    expect_rejected(f.config, f.header, 0, -2);
    expect_rejected(f.config, f.header, -1, 2);
    expect_rejected(f.config, f.header, 2, 2);
    expect_rejected(f.config, f.header, 4, 4);
    expect_rejected(f.config, f.header, 0, 3);
    expect_rejected(f.config, f.header, 0, 8);
}

TEST(TensorParallel, RejectsAnMlpThatDoesNotDivide) {
    ModelConfig c = tp_config();
    c.intermediate = 26;
    const SafetensorsHeader h = fake_header(c);
    EXPECT_NO_THROW(plan_shard_layout(c, h, 1, 2));
    expect_rejected(c, h, 0, 4);
}

TEST(TensorParallel, RejectsWhatPlanWeightLayoutRejects) {
    const ModelConfig c = tp_config();
    SafetensorsHeader missing = fake_header(c);
    missing.tensors.erase("model.layers.1.mlp.down_proj.weight");
    expect_runtime_error(c, missing, "missing tensor model.layers.1.mlp.down_proj.weight");

    SafetensorsHeader wrong_shape = fake_header(c);
    wrong_shape.tensors.at("model.layers.0.self_attn.o_proj.weight").shape = {32, 16};
    expect_runtime_error(c, wrong_shape, "self_attn.o_proj.weight has shape");

    SafetensorsHeader wrong_type = fake_header(c);
    wrong_type.tensors.at("model.norm.weight").dtype = engine::DType::F32;
    expect_runtime_error(c, wrong_type, "model.norm.weight must be BF16");

    SafetensorsHeader extra = fake_header(c);
    extra.tensors["model.extra.weight"] = {engine::DType::BF16, {4}, 0, 8};
    expect_runtime_error(c, extra, "unexpected tensor model.extra.weight");

    ModelConfig tied = c;
    tied.tie_embeddings = true;
    expect_runtime_error(tied, fake_header(c), "tied embeddings");
}

}  // namespace
