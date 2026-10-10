#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "engine/runtime/sampler.h"
#include "engine/sampling/greedy.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Sampler;
using engine::SampleRow;
using engine::SamplingParams;
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

class Batch {
public:
    Batch(const std::vector<uint16_t>& logits, int64_t vocab)
        : rows_(int64_t(logits.size()) / vocab),
          vocab_(vocab),
          logits_(upload(logits)),
          ids_(size_t(rows_) * sizeof(int32_t)),
          sampler_(rows_) {}

    std::vector<int32_t> operator()(const std::vector<SampleRow>& rows) {
        sampler_(ids(), logits(), rows, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return download<int32_t>(ids_);
    }

    std::vector<int32_t> operator()(const SamplingParams& params, uint64_t first_step) {
        std::vector<SampleRow> rows(static_cast<size_t>(rows_));
        for (size_t r = 0; r < rows.size(); ++r) rows[r] = {params, first_step + r};
        return (*this)(rows);
    }

    std::vector<int32_t> greedy() {
        engine::greedy(ids(), logits(), stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return download<int32_t>(ids_);
    }

    int64_t rows() const { return rows_; }

private:
    Tensor ids() { return Tensor(ids_.data(), DType::I32, {rows_}); }
    Tensor logits() { return Tensor(logits_.data(), DType::BF16, {rows_, vocab_}); }

    int64_t rows_;
    int64_t vocab_;
    CudaStream stream_;
    DeviceBuffer logits_;
    DeviceBuffer ids_;
    Sampler sampler_;
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

std::vector<int64_t> histogram(Batch& batch, int64_t vocab, const SamplingParams& params,
                               int draws) {
    std::vector<int64_t> counts(size_t(vocab), 0);
    for (int d = 0; d < draws; ++d)
        for (const auto id : batch(params, uint64_t(d * batch.rows()))) ++counts[size_t(id)];
    return counts;
}

}  // namespace

TEST(Sample, MatchesSoftmaxOnASmallVocab) {
    std::vector<uint16_t> row;
    for (const float v : {2.0f, 1.0f, 0.0f, -1.0f, 0.5f, -INFINITY, -3.0f})
        row.push_back(to_bf16(v));
    const auto vocab = int64_t(row.size());
    Batch batch(repeat_rows(row, 50'000), vocab);
    for (const float temperature : {0.25f, 0.5f, 1.0f, 2.0f}) {
        SCOPED_TRACE("temperature " + std::to_string(temperature));
        const auto counts = histogram(batch, vocab, {temperature, 7}, 4);
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
        Batch batch(repeat_rows(row, 1000), vocab);
        for (const float temperature : {0.7f, 1.0f, 1.5f}) {
            SCOPED_TRACE("temperature " + std::to_string(temperature));
            const auto counts = histogram(batch, vocab, {temperature, 7}, 100);
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
    Batch batch(logits, vocab);
    const auto greedy = batch.greedy();
    for (uint64_t step = 0; step < 4; ++step)
        EXPECT_EQ(batch({0.05f, 1}, step * uint64_t(rows)), greedy);
}

TEST(Sample, TemperatureZeroIsTheGreedyKernelWhateverTheSeedAndStep) {
    const int64_t vocab = 32000;
    const int64_t rows = 64;
    Batch batch(engine::test::random_bf16(size_t(rows * vocab), 11, -2, 2), vocab);
    const auto greedy = batch.greedy();
    EXPECT_EQ(batch({0.0f, 0}, 0), greedy);
    EXPECT_EQ(batch({0.0f, 99}, 12345), greedy);
}

TEST(Sample, TheSameSeedAndStepRepeatAndAnyChangeRedraws) {
    const int64_t vocab = 32000;
    const int64_t rows = 256;
    Batch batch(std::vector<uint16_t>(size_t(rows * vocab), to_bf16(0)), vocab);
    const auto base = batch({1, 42}, 5);
    EXPECT_EQ(batch({1, 42}, 5), base);
    const auto differences = [&](const std::vector<int32_t>& other) {
        int64_t n = 0;
        for (size_t r = 0; r < base.size(); ++r) n += base[r] != other[r];
        return n;
    };
    EXPECT_GE(differences(batch({1, 42}, 5 + rows)), rows - 2);
    EXPECT_GE(differences(batch({1, 43}, 5)), rows - 2);
    EXPECT_GE(int64_t(std::set<int32_t>(base.begin(), base.end()).size()), rows - 2);
}

TEST(Sample, ARowDependsOnlyOnItsLogitsAndParametersNotOnItsBatch) {
    const int64_t vocab = 32000;
    const int64_t rows = 8;
    const auto logits = engine::test::random_bf16(size_t(rows * vocab), 5, -4, 4);
    const std::vector<SampleRow> params = {
        {{0.0f, 1}, 0},  {{1.0f, 1}, 0}, {{1.0f, 1}, 1},   {{1.0f, 2}, 0},
        {{0.7f, 3}, 17}, {{1.5f, 4}, 9}, {{0.0f, 5}, 100}, {{1.0f, 1}, 0},
    };
    Batch all(logits, vocab);
    const auto together = all(params);

    for (int64_t r = 0; r < rows; ++r) {
        SCOPED_TRACE("row " + std::to_string(r));
        const auto first = logits.begin() + r * vocab;
        Batch alone(std::vector<uint16_t>(first, first + vocab), vocab);
        EXPECT_EQ(alone({params[size_t(r)]}), std::vector<int32_t>{together[size_t(r)]});
    }

    std::vector<uint16_t> reversed_logits;
    for (int64_t r = rows - 1; r >= 0; --r)
        reversed_logits.insert(reversed_logits.end(), logits.begin() + r * vocab,
                               logits.begin() + (r + 1) * vocab);
    Batch reversed(reversed_logits, vocab);
    auto out = reversed(std::vector<SampleRow>(params.rbegin(), params.rend()));
    std::reverse(out.begin(), out.end());
    EXPECT_EQ(out, together);
}

TEST(Sample, ARowOfNegativeInfinityPicksTheFirstIndex) {
    Batch batch(std::vector<uint16_t>(3000, to_bf16(-INFINITY)), 3000);
    EXPECT_EQ(batch({1.0f, 0}, 0), (std::vector<int32_t>{0}));
    EXPECT_EQ(batch({0.0f, 0}, 0), (std::vector<int32_t>{0}));
}

TEST(Sample, ZeroRowsIsANoOp) {
    CudaStream stream;
    Sampler sampler(1);
    EXPECT_NO_THROW(sampler(Tensor(nullptr, DType::I32, {0}),
                            Tensor(nullptr, DType::BF16, {0, 32000}), {}, stream));
}

TEST(Sample, RejectsBadArguments) {
    EXPECT_THROW(Sampler(0), std::invalid_argument);
    EXPECT_THROW(Sampler(-1), std::invalid_argument);

    CudaStream stream;
    Sampler sampler(2);
    int dummy = 0;
    void* p = &dummy;
    const Tensor ids(p, DType::I32, {2});
    const Tensor logits(p, DType::BF16, {2, 10});
    const std::vector<SampleRow> two(2, SampleRow{{1.0f, 0}, 0});
    const std::vector<SampleRow> three(3, SampleRow{{1.0f, 0}, 0});

    EXPECT_THROW(sampler(ids, logits, std::vector<SampleRow>(1), stream), std::invalid_argument);
    EXPECT_THROW(
        sampler(Tensor(p, DType::I32, {3}), Tensor(p, DType::BF16, {3, 10}), three, stream),
        std::invalid_argument);
    for (const float t : {-1.0f, INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        auto bad = two;
        bad[1].params.temperature = t;
        EXPECT_THROW(sampler(ids, logits, bad, stream), std::invalid_argument);
    }
    EXPECT_THROW(sampler(ids, Tensor(p, DType::F32, {2, 10}), two, stream), std::invalid_argument);
    EXPECT_THROW(sampler(ids, Tensor(p, DType::BF16, {2, 0}), two, stream), std::invalid_argument);
    EXPECT_THROW(sampler(ids, Tensor(p, DType::BF16, {20}), two, stream), std::invalid_argument);
    EXPECT_THROW(sampler(Tensor(p, DType::I32, {3}), logits, two, stream), std::invalid_argument);
    EXPECT_THROW(sampler(Tensor(p, DType::I64, {2}), logits, two, stream), std::invalid_argument);
}
