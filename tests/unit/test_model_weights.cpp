#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "engine/loader/weight_layout.h"
#include "engine/model/weights.h"
#include "support/fake_checkpoint.h"

using engine::DType;
using engine::Region;
using engine::Tensor;

namespace {

void expect_view(const Tensor& t, const Region& r, std::byte* base) {
    EXPECT_EQ(t.data, base + r.offset);
    EXPECT_EQ(t.dtype, DType::BF16);
    ASSERT_EQ(size_t(t.ndim), r.shape.size());
    for (int i = 0; i < t.ndim; ++i) EXPECT_EQ(t.shape[i], r.shape[i]);
    EXPECT_EQ(t.bytes(), r.bytes);
}

}  // namespace

TEST(ModelWeights, ViewsPointIntoTheBaseAllocation) {
    const auto c = engine::test::tinyllama_config();
    const auto layout = engine::plan_weight_layout(c, engine::test::fake_header(c));
    std::vector<std::byte> fake_device(16);
    std::byte* base = fake_device.data();

    const auto w = engine::bind_weights(layout, base);
    expect_view(w.embed, layout.embed, base);
    expect_view(w.final_norm, layout.final_norm, base);
    expect_view(w.lm_head, layout.lm_head, base);
    ASSERT_EQ(w.layers.size(), 22u);
    for (size_t i = 0; i < w.layers.size(); ++i) {
        SCOPED_TRACE(i);
        expect_view(w.layers[i].ln1, layout.layers[i].ln1, base);
        expect_view(w.layers[i].wqkv, layout.layers[i].wqkv, base);
        expect_view(w.layers[i].wo, layout.layers[i].wo, base);
        expect_view(w.layers[i].ln2, layout.layers[i].ln2, base);
        expect_view(w.layers[i].wgu, layout.layers[i].wgu, base);
        expect_view(w.layers[i].wdown, layout.layers[i].wdown, base);
    }
    EXPECT_EQ(w.layers[0].wqkv.shape[0], 2560);
    EXPECT_EQ(w.layers[0].wgu.shape[0], 11264);
}
