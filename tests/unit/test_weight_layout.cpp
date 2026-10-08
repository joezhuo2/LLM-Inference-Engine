#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loader/weight_layout.h"
#include "support/fake_checkpoint.h"

using engine::CopyOp;
using engine::DType;
using engine::kWeightAlignment;
using engine::plan_weight_layout;
using engine::Region;
using engine::SafetensorsHeader;
using engine::WeightLayout;
using engine::test::fake_header;
using engine::test::tiny_config;
using engine::test::tinyllama_config;

namespace {

std::vector<Region> regions(const WeightLayout& w) {
    std::vector<Region> r = {w.embed};
    for (const auto& l : w.layers) r.insert(r.end(), {l.ln1, l.wqkv, l.wo, l.ln2, l.wgu, l.wdown});
    r.push_back(w.final_norm);
    r.push_back(w.lm_head);
    return r;
}

CopyOp copy_of(const WeightLayout& w, const std::string& name) {
    const auto it = std::find_if(w.copies.begin(), w.copies.end(),
                                 [&](const CopyOp& c) { return c.name == name; });
    if (it == w.copies.end()) throw std::out_of_range(name);
    return *it;
}

void expect_rejected(const SafetensorsHeader& h, const std::string& fragment,
                     engine::ModelConfig c = tiny_config()) {
    try {
        plan_weight_layout(c, h);
        ADD_FAILURE() << "expected a rejection mentioning '" << fragment << "'";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

}  // namespace

TEST(WeightLayout, CopiesEveryTensorExactlyOnce) {
    const auto h = fake_header(tiny_config());
    const WeightLayout w = plan_weight_layout(tiny_config(), h);
    ASSERT_EQ(w.copies.size(), h.tensors.size());
    std::set<std::string> names;
    for (const CopyOp& c : w.copies) {
        EXPECT_TRUE(names.insert(c.name).second) << c.name;
        EXPECT_EQ(c.src, h.tensors.at(c.name).offset) << c.name;
        EXPECT_EQ(c.bytes, h.tensors.at(c.name).bytes) << c.name;
    }
}

TEST(WeightLayout, RegionsAreAlignedOrderedAndDisjoint) {
    const WeightLayout w = plan_weight_layout(tiny_config(), fake_header(tiny_config()));
    const auto r = regions(w);
    for (size_t i = 0; i < r.size(); ++i) {
        EXPECT_EQ(r[i].offset % kWeightAlignment, 0u) << "region " << i;
        if (i + 1 < r.size()) {
            EXPECT_LE(r[i].offset + r[i].bytes, r[i + 1].offset) << "region " << i;
        }
    }
    EXPECT_EQ(w.total_bytes, r.back().offset + r.back().bytes);
}

TEST(WeightLayout, EveryCopyLandsInsideItsRegion) {
    const WeightLayout w = plan_weight_layout(tiny_config(), fake_header(tiny_config()));
    const auto r = regions(w);
    uint64_t covered = 0;
    for (const CopyOp& c : w.copies) {
        const bool inside = std::any_of(r.begin(), r.end(), [&](const Region& g) {
            return c.dst >= g.offset && c.dst + c.bytes <= g.offset + g.bytes;
        });
        EXPECT_TRUE(inside) << c.name;
        covered += c.bytes;
    }
    uint64_t region_bytes = 0;
    for (const Region& g : r) region_bytes += g.bytes;
    EXPECT_EQ(covered, region_bytes);
}

TEST(WeightLayout, FusesQkvAsQThenKThenV) {
    const auto c = tiny_config();
    const WeightLayout w = plan_weight_layout(c, fake_header(c));
    const auto& l = w.layers[1];
    EXPECT_EQ(l.wqkv.shape, (std::vector<int64_t>{c.qkv_dim(), c.hidden}));
    const uint64_t row = uint64_t(c.hidden) * 2;
    EXPECT_EQ(copy_of(w, "model.layers.1.self_attn.q_proj.weight").dst, l.wqkv.offset);
    EXPECT_EQ(copy_of(w, "model.layers.1.self_attn.k_proj.weight").dst,
              l.wqkv.offset + uint64_t(c.hidden) * row);
    const CopyOp v = copy_of(w, "model.layers.1.self_attn.v_proj.weight");
    EXPECT_EQ(v.dst, l.wqkv.offset + uint64_t(c.hidden + c.kv_dim()) * row);
    EXPECT_EQ(v.dst + v.bytes, l.wqkv.offset + l.wqkv.bytes);
}

TEST(WeightLayout, FusesGateThenUp) {
    const auto c = tiny_config();
    const WeightLayout w = plan_weight_layout(c, fake_header(c));
    const auto& l = w.layers[0];
    EXPECT_EQ(l.wgu.shape, (std::vector<int64_t>{2 * c.intermediate, c.hidden}));
    EXPECT_EQ(copy_of(w, "model.layers.0.mlp.gate_proj.weight").dst, l.wgu.offset);
    EXPECT_EQ(copy_of(w, "model.layers.0.mlp.up_proj.weight").dst,
              l.wgu.offset + uint64_t(c.intermediate) * c.hidden * 2);
    EXPECT_EQ(l.wdown.shape, (std::vector<int64_t>{c.hidden, c.intermediate}));
}

TEST(WeightLayout, CopiesAreSortedBySourceOffset) {
    const WeightLayout w = plan_weight_layout(tiny_config(), fake_header(tiny_config(), 4096));
    EXPECT_TRUE(std::is_sorted(w.copies.begin(), w.copies.end(),
                               [](const CopyOp& a, const CopyOp& b) { return a.src < b.src; }));
    EXPECT_EQ(w.copies.front().src, 4096u);
}

TEST(WeightLayout, TinyLlamaNeedsNoPadding) {
    const auto c = tinyllama_config();
    const auto h = fake_header(c);
    const WeightLayout w = plan_weight_layout(c, h);
    EXPECT_EQ(w.copies.size(), 201u);
    EXPECT_EQ(w.layers.size(), 22u);
    EXPECT_EQ(w.layers[0].wqkv.shape, (std::vector<int64_t>{2560, 2048}));
    EXPECT_EQ(w.layers[0].wgu.shape, (std::vector<int64_t>{11264, 2048}));
    EXPECT_EQ(w.total_bytes, 2'200'096'768u);
}

TEST(WeightLayout, RejectsMissingTensor) {
    auto h = fake_header(tiny_config());
    h.tensors.erase("model.layers.1.mlp.up_proj.weight");
    expect_rejected(h, "model.layers.1.mlp.up_proj.weight");
}

TEST(WeightLayout, RejectsWrongShape) {
    auto h = fake_header(tiny_config());
    h.tensors.at("model.layers.0.self_attn.k_proj.weight").shape = {8, 8};
    expect_rejected(h, "model.layers.0.self_attn.k_proj.weight has shape [8, 8], expected [4, 8]");
}

TEST(WeightLayout, RejectsNonBf16) {
    auto h = fake_header(tiny_config());
    h.tensors.at("model.norm.weight").dtype = DType::F32;
    expect_rejected(h, "model.norm.weight must be BF16");
}

TEST(WeightLayout, RejectsUnexpectedTensor) {
    auto h = fake_header(tiny_config());
    h.tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = {DType::BF16, {2}, 0, 4};
    expect_rejected(h, "unexpected tensor model.layers.0.self_attn.rotary_emb.inv_freq");
}

TEST(WeightLayout, RejectsTiedEmbeddings) {
    auto c = tiny_config();
    c.tie_embeddings = true;
    expect_rejected(fake_header(c), "tied embeddings", c);
}
