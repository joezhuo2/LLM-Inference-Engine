#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/runtime/model_runner.h"
#include "support/device.h"
#include "support/logits.h"
#include "support/model.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::ModelRunner;
using engine::Tensor;
using engine::test::from_bf16;

namespace {

constexpr double kLayerRelativeL2 = 0.05;
constexpr double kMeanAbsLogitError = 0.05;
constexpr double kArgmaxMatch = 0.99;

std::vector<uint16_t> to_host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

}  // namespace

TEST(ForwardGolden, TeacherForcedMatchesHuggingFaceOnEveryPrompt) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto& w = engine::test::tinyllama();
    const int64_t vocab = w.config.vocab;
    engine::test::CudaStream stream;
    ModelRunner runner(w.config, w.weights, 512, 2048, stream);

    size_t positions = 0, matches = 0;
    double abs_error = 0.0;
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const auto ids = ref.read<int32_t>("input_ids");
        double worst = 0.0;
        int worst_layer = 0;
        runner.reset();
        const auto logits = to_host(runner.forward(ids, true, [&](int layer, const Tensor& hidden) {
            const double rel = engine::test::relative_l2(
                to_host(hidden), ref.read<uint16_t>("layers." + std::to_string(layer) + ".out"));
            EXPECT_LE(rel, kLayerRelativeL2) << "layer " << layer;
            if (rel > worst) {
                worst = rel;
                worst_layer = layer;
            }
        }));
        const auto expected = ref.read<uint16_t>("logits");
        const size_t match = engine::test::count_equal(engine::test::argmax_rows(logits, vocab),
                                                       engine::test::argmax_rows(expected, vocab));
        const double mean_abs = engine::test::mean_abs_diff(logits, expected);
        std::printf(
            "%-16s %4zu tokens  argmax %4zu/%-4zu  mean |dlogit| %.4f  worst layer %2d (%.4f)\n",
            prompt.name.c_str(), ids.size(), match, ids.size(), mean_abs, worst_layer, worst);
        positions += ids.size();
        matches += match;
        abs_error += mean_abs * double(ids.size()) * double(vocab);
    }
    const double match_rate = double(matches) / double(positions);
    const double mean_abs = abs_error / (double(positions) * double(vocab));
    std::printf("teacher-forced argmax %zu/%zu (%.2f%%), mean |dlogit| %.4f\n", matches, positions,
                100.0 * match_rate, mean_abs);
    EXPECT_GE(match_rate, kArgmaxMatch);
    EXPECT_LT(mean_abs, kMeanAbsLogitError);
}
