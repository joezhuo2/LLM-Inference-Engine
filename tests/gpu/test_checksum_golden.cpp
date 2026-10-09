#include <gtest/gtest.h>

#include <filesystem>

#include "engine/runtime/checksum.h"
#include "engine/runtime/weights.h"
#include "kernels/cuda_check.h"
#include "support/checksum_golden.h"
#include "support/paths.h"

TEST(ChecksumGolden, DeviceWeightsMatchPyTorch) {
    const auto dir = engine::test::model_dir();
    const auto golden = engine::test::golden_checksums_path();
    if (!std::filesystem::exists(dir / "model.safetensors")) GTEST_SKIP() << dir << " not found";
    if (!std::filesystem::exists(golden))
        GTEST_SKIP() << golden << " not found, run scripts/checksum.py";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    {
        const engine::DeviceWeights w = engine::load_weights(dir, stream);
        engine::test::expect_matches_golden(engine::checksum_device(w.layout, w.buffer.data()),
                                            engine::test::load_golden_checksums());
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
