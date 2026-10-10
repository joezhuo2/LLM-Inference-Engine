#include "engine/runtime/sampler.h"

#include <stdexcept>

#include "kernels/cuda_check.h"

namespace engine {

Sampler::Sampler(int64_t max_rows) : max_rows_(max_rows) {
    if (max_rows < 1) throw std::invalid_argument("Sampler: max_rows must be positive");
    rows_ = DeviceBuffer(size_t(max_rows) * sizeof(SampleRow));
}

void Sampler::operator()(const Tensor& ids, const Tensor& logits, std::span<const SampleRow> rows,
                         Stream stream) {
    if (logits.ndim != 2 || int64_t(rows.size()) != logits.shape[0])
        throw std::invalid_argument("Sampler: needs one SampleRow per row of logits");
    if (int64_t(rows.size()) > max_rows_)
        throw std::invalid_argument("Sampler: more rows than max_rows");
    for (const auto& row : rows) validate(row.params);
    if (!rows.empty())
        CUDA_CHECK(cudaMemcpyAsync(rows_.data(), rows.data(), rows.size_bytes(),
                                   cudaMemcpyHostToDevice, stream));
    sample(ids, logits, static_cast<const SampleRow*>(rows_.data()), stream);
}

}  // namespace engine
