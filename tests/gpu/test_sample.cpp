#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "engine/sampling/greedy.h"
#include "engine/sampling/sample.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::sample;
using engine::Tensor;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

std::vector<uint16_t> repeat_rows(const std::vector<uint16_t>& row, int64_t rows) {
    std::vector<uint16_t> out;
    out.reserve(row.size() * size_t(rows));
    for (int64_t r = 0; r < rows; ++r) out.insert(out.end(), row.begin(), row.end());
    return out;
}

class Sampler {
public:
    Sampler(const std::vector<uint16_t>& logits, int64_t vocab)
        : rows_(int64_t(logits.size()) / vocab),
          vocab_(vocab),
          logits_(upload(logits)),
          ids_(size_t(rows_) * sizeof(int32_t)) {}

    std::vector<int32_t> operator()(float temperature, uint64_t seed, uint64_t step) {
        sample(ids(), logits(), temperature, seed, step, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return download<int32_t>(ids_);
    }

    std::vector<int32_t> greedy() {
        engine::greedy(ids(), logits(), stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return download<int32_t>(ids_);
    }

private:
    Tensor ids() { return Tensor(ids_.data(), DType::I32, {rows_}); }
    Tensor logits() { return Tensor(logits_.data(), DType::BF16, {rows_, vocab_}); }

    int64_t rows_;
    int64_t vocab_;
    CudaStream stream_;
    DeviceBuffer logits_;
    DeviceBuffer ids_;
};

std::vector<double> softmax(const std::vector<uint16_t>& row, float temperature) {
    std::vector<double> p(row.size());
    double top = -INFINITY;
    for (size_t i = 0; i < row.size(); ++i) {
        p[i] = double(from_bf16(row[i]) / temperature);
        top = std::max(top, p[i]);
    }
    double sum = 0;
    for (auto& v : p) sum += v = std::exp(v - top);
    for (auto& v : p) v /= sum;
    return p;
}

double chi_square_critical(int dof) {
    const double k = dof;
    const double z = 3.090;
    return k * std::pow(1 - 2 / (9 * k) + z * std::sqrt(2 / (9 * k)), 3);
}

struct Fit {
    double chi2 = 0;
    double critical = 0;
    int64_t impossible = 0;
};

Fit goodness_of_fit(const std::vector<int64_t>& counts, const std::vector<double>& p,
                    double min_expected) {
    int64_t n = 0;
    for (const auto c : counts) n += c;
    Fit fit;
    double rest_expected = 0;
    int64_t rest_count = 0;
    int buckets = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double expected = p[i] * double(n);
        if (p[i] == 0) {
            fit.impossible += counts[i];
        } else if (expected >= min_expected) {
            fit.chi2 += std::pow(double(counts[i]) - expected, 2) / expected;
            ++buckets;
        } else {
            rest_expected += expected;
            rest_count += counts[i];
        }
    }
    if (rest_expected > 0) {
        fit.chi2 += std::pow(double(rest_count) - rest_expected, 2) / rest_expected;
        ++buckets;
    }
    fit.critical = chi_square_critical(buckets - 1);
    return fit;
}

std::vector<int64_t> histogram(Sampler& sampler, int64_t vocab, float temperature, int steps) {
    std::vector<int64_t> counts(size_t(vocab), 0);
    for (int step = 0; step < steps; ++step)
        for (const auto id : sampler(temperature, 7, uint64_t(step))) ++counts[size_t(id)];
    return counts;
}

}  // namespace

TEST(Sample, MatchesSoftmaxOnASmallVocab) {
    std::vector<uint16_t> row;
    for (const float v : {2.0f, 1.0f, 0.0f, -1.0f, 0.5f, -INFINITY, -3.0f})
        row.push_back(to_bf16(v));
    const auto vocab = int64_t(row.size());
    Sampler sampler(repeat_rows(row, 50'000), vocab);
    for (const float temperature : {0.25f, 0.5f, 1.0f, 2.0f}) {
        SCOPED_TRACE("temperature " + std::to_string(temperature));
        const auto counts = histogram(sampler, vocab, temperature, 4);
        const Fit fit = goodness_of_fit(counts, softmax(row, temperature), 5);
        std::printf("temperature %.2f: chi-square %.2f, critical %.2f\n", temperature, fit.chi2,
                    fit.critical);
        EXPECT_EQ(fit.impossible, 0);
        EXPECT_LT(fit.chi2, fit.critical);
    }
}

TEST(Sample, MatchesSoftmaxOnHuggingFaceLogits) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto prompts = engine::test::load_reference_manifest();
    for (const auto& prompt : {prompts.front(), prompts.back()}) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const int64_t vocab = ref.entry("logits").shape[1];
        const auto logits = ref.read<uint16_t>("logits");
        const auto first = logits.begin() + (prompt.prompt_len - 1) * vocab;
        const std::vector<uint16_t> row(first, first + vocab);
        Sampler sampler(repeat_rows(row, 1000), vocab);
        for (const float temperature : {0.7f, 1.0f, 1.5f}) {
            SCOPED_TRACE("temperature " + std::to_string(temperature));
            const auto counts = histogram(sampler, vocab, temperature, 100);
            const Fit fit = goodness_of_fit(counts, softmax(row, temperature), 20);
            std::printf("%s, temperature %.1f: chi-square %.2f, critical %.2f\n",
                        prompt.name.c_str(), temperature, fit.chi2, fit.critical);
            EXPECT_LT(fit.chi2, fit.critical);
        }
    }
}

TEST(Sample, LowTemperatureIsGreedy) {
    const int64_t vocab = 32000;
    const int64_t rows = 64;
    auto logits = engine::test::random_bf16(size_t(rows * vocab), 3, -8, 4);
    for (int64_t r = 0; r < rows; ++r) logits[size_t(r * vocab + (r * 997) % vocab)] = to_bf16(6);
    Sampler sampler(logits, vocab);
    const auto greedy = sampler.greedy();
    for (uint64_t step = 0; step < 4; ++step) EXPECT_EQ(sampler(0.05f, 1, step), greedy);
}

TEST(Sample, TheSameSeedAndStepRepeatAndAnyChangeDraws) {
    const int64_t vocab = 32000;
    const int64_t rows = 256;
    Sampler sampler(std::vector<uint16_t>(size_t(rows * vocab), to_bf16(0)), vocab);
    const auto base = sampler(1, 42, 5);
    EXPECT_EQ(sampler(1, 42, 5), base);
    const auto differences = [&](const std::vector<int32_t>& other) {
        int64_t n = 0;
        for (size_t r = 0; r < base.size(); ++r) n += base[r] != other[r];
        return n;
    };
    EXPECT_GE(differences(sampler(1, 42, 6)), rows - 2);
    EXPECT_GE(differences(sampler(1, 43, 5)), rows - 2);
    EXPECT_GE(int64_t(std::set<int32_t>(base.begin(), base.end()).size()), rows - 2);
}

TEST(Sample, ARowOfNegativeInfinityPicksTheFirstIndex) {
    Sampler sampler(std::vector<uint16_t>(3000, to_bf16(-INFINITY)), 3000);
    EXPECT_EQ(sampler(1, 0, 0), (std::vector<int32_t>{0}));
}

TEST(Sample, ZeroTokensIsANoOp) {
    CudaStream stream;
    EXPECT_NO_THROW(sample(Tensor(nullptr, DType::I32, {0}),
                           Tensor(nullptr, DType::BF16, {0, 32000}), 1, 0, 0, stream));
}

TEST(Sample, RejectsBadArguments) {
    CudaStream stream;
    int dummy = 0;
    void* p = &dummy;
    const Tensor ids(p, DType::I32, {2});
    const Tensor logits(p, DType::BF16, {2, 10});
    for (const float t : {0.0f, -1.0f, INFINITY, std::numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(sample(ids, logits, t, 0, 0, stream), std::invalid_argument);
    EXPECT_THROW(sample(ids, Tensor(p, DType::F32, {2, 10}), 1, 0, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(sample(ids, Tensor(p, DType::BF16, {2, 0}), 1, 0, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(sample(Tensor(p, DType::I32, {3}), logits, 1, 0, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(sample(Tensor(p, DType::I64, {2}), logits, 1, 0, 0, stream),
                 std::invalid_argument);
}
