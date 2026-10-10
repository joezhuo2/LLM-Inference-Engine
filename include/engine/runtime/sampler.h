#pragma once

#include <cstdint>
#include <span>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/runtime/device_buffer.h"
#include "engine/sampling/sample.h"

namespace engine {

class Sampler {
public:
    explicit Sampler(int64_t max_rows);

    void operator()(const Tensor& ids, const Tensor& logits, std::span<const SampleRow> rows,
                    Stream stream);

    int64_t max_rows() const { return max_rows_; }

private:
    int64_t max_rows_;
    DeviceBuffer rows_;
};

}  // namespace engine
