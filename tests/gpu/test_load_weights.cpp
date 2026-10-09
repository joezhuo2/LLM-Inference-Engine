#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/loader/mapped_file.h"
#include "engine/runtime/weights.h"
#include "kernels/cuda_check.h"
#include "support/paths.h"

using engine::CopyOp;

TEST(LoadWeights, RealTinyLlamaLandsOnTheDevice) {
    const auto dir = engine::test::model_dir();
    if (!std::filesystem::exists(dir / "model.safetensors")) GTEST_SKIP() << dir << " not found";
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    {
        const engine::DeviceWeights w = engine::load_weights(dir, stream);
        EXPECT_EQ(w.config.layers, 22);
        EXPECT_EQ(w.layout.copies.size(), 201u);
        EXPECT_EQ(w.layout.total_bytes, 2'200'096'768u);
        EXPECT_EQ(w.stats.bytes, 2'200'096'768u);
        EXPECT_GT(w.stats.gb_per_s(), 0.0);
        ASSERT_EQ(w.weights.layers.size(), 22u);
        EXPECT_EQ(w.weights.layers[21].wqkv.shape[0], 2560);
        EXPECT_EQ(w.weights.lm_head.shape[0], 32000);

        const engine::MappedFile file(dir / "model.safetensors");
        for (const std::string name :
             {"model.layers.0.self_attn.q_proj.weight", "model.layers.21.self_attn.v_proj.weight",
              "model.norm.weight", "lm_head.weight"}) {
            const auto it = std::find_if(w.layout.copies.begin(), w.layout.copies.end(),
                                         [&](const CopyOp& c) { return c.name == name; });
            ASSERT_NE(it, w.layout.copies.end()) << name;
            std::vector<std::byte> back(it->bytes);
            CUDA_CHECK(cudaMemcpy(back.data(),
                                  static_cast<const std::byte*>(w.buffer.data()) + it->dst,
                                  back.size(), cudaMemcpyDeviceToHost));
            EXPECT_EQ(std::memcmp(back.data(), file.bytes().data() + it->src, back.size()), 0)
                << name;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
