#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/add.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/model/config.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/paths.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::add;
using engine::test::close_up_to_reduction_order;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

std::vector<uint16_t> cpu_add(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    std::vector<uint16_t> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = to_bf16(from_bf16(a[i]) + from_bf16(b[i]));
    return out;
}

}  // namespace

TEST(Add, MatchesCpuReference) {
    for (const int64_t n : {1, 255, 256, 257, 100'000}) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto a = engine::test::random_bf16(size_t(n), 7, -8.0f, 8.0f);
        const auto b = engine::test::random_bf16(size_t(n), 8, -0.05f, 0.05f);
        CudaStream stream;
        DeviceBuffer da = upload(a), db = upload(b), out(a.size() * sizeof(uint16_t));
        add(Tensor(out.data(), DType::BF16, {n}), Tensor(da.data(), DType::BF16, {n}),
            Tensor(db.data(), DType::BF16, {n}), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto diff = compare_bf16(download<uint16_t>(out), cpu_add(a, b));
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}

TEST(Add, WorksInPlace) {
    const int64_t tokens = 3, hidden = 2048;
    const auto a = engine::test::random_bf16(size_t(tokens * hidden), 9);
    const auto b = engine::test::random_bf16(size_t(tokens * hidden), 10);
    CudaStream stream;
    DeviceBuffer da = upload(a), db = upload(b);
    const Tensor t(da.data(), DType::BF16, {tokens, hidden});
    add(t, t, Tensor(db.data(), DType::BF16, {tokens, hidden}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto diff = compare_bf16(download<uint16_t>(da), cpu_add(a, b));
    EXPECT_EQ(diff.mismatches, 0u) << diff;
}

TEST(Add, ZeroElementsIsANoOp) {
    CudaStream stream;
    const Tensor empty(nullptr, DType::BF16, {0, 2048});
    add(empty, empty, empty, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Add, RejectsMismatchedShapesAndDtypes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    EXPECT_THROW(add(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::BF16, {3, 8}),
                     Tensor(p, DType::BF16, {8, 3}), stream),
                 std::invalid_argument);
    EXPECT_THROW(add(Tensor(p, DType::BF16, {24}), Tensor(p, DType::BF16, {3, 8}),
                     Tensor(p, DType::BF16, {3, 8}), stream),
                 std::invalid_argument);
    EXPECT_THROW(add(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::F32, {3, 8}),
                     Tensor(p, DType::BF16, {3, 8}), stream),
                 std::invalid_argument);
}

TEST(Add, RebuildsHuggingFaceLn2AndOutOnEveryTracedLayer) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const engine::test::SafetensorsFile weights(engine::test::model_dir() / "model.safetensors");
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    const int64_t tokens = ref.entry("embed").shape[0], hidden = config.hidden;
    const auto shape = {tokens, hidden};
    CudaStream stream;
    for (int layer = 0; layer < config.layers; ++layer) {
        SCOPED_TRACE("layer " + std::to_string(layer));
        const std::string l = "layers." + std::to_string(layer) + ".";
        const std::string in =
            layer == 0 ? "embed" : "layers." + std::to_string(layer - 1) + ".out";
        DeviceBuffer x = upload(ref.read<uint16_t>(in)), o = upload(ref.read<uint16_t>(l + "o"));
        DeviceBuffer down = upload(ref.read<uint16_t>(l + "down"));
        DeviceBuffer w =
            upload(weights.read<uint16_t>("model." + l + "post_attention_layernorm.weight"));
        DeviceBuffer h(x.size()), ln2(x.size()), out(x.size());

        add(Tensor(h.data(), DType::BF16, shape), Tensor(x.data(), DType::BF16, shape),
            Tensor(o.data(), DType::BF16, shape), stream);
        engine::naive::rmsnorm(
            Tensor(ln2.data(), DType::BF16, shape), Tensor(h.data(), DType::BF16, shape),
            Tensor(w.data(), DType::BF16, {hidden}), float(config.rms_eps), stream);
        add(Tensor(out.data(), DType::BF16, shape), Tensor(h.data(), DType::BF16, shape),
            Tensor(down.data(), DType::BF16, shape), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto ln2_diff = compare_bf16(download<uint16_t>(ln2), ref.read<uint16_t>(l + "ln2"));
        EXPECT_TRUE(close_up_to_reduction_order(ln2_diff, size_t(tokens * hidden)))
            << "ln2: " << ln2_diff;
        const auto out_diff = compare_bf16(download<uint16_t>(out), ref.read<uint16_t>(l + "out"));
        EXPECT_EQ(out_diff.mismatches, 0u) << "out: " << out_diff;
    }
}
