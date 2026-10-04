#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace engine {

enum class DType : uint8_t { BF16, F16, F32, I32, I64 };

constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::BF16:
        case DType::F16:
            return 2;
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::I64:
            return 8;
    }
    return 0;
}

// Non-owning view of contiguous row-major memory, usually on the device.
struct Tensor {
    static constexpr int kMaxDims = 4;

    void* data = nullptr;
    DType dtype = DType::BF16;
    std::array<int64_t, kMaxDims> shape{};
    int ndim = 0;

    Tensor() = default;
    Tensor(void* data, DType dtype, std::initializer_list<int64_t> dims);

    int64_t numel() const;
    size_t bytes() const { return size_t(numel()) * dtype_size(dtype); }
};

}  // namespace engine
