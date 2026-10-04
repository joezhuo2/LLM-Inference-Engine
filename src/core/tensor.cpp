#include "engine/core/tensor.h"

#include <algorithm>
#include <stdexcept>

namespace engine {

Tensor::Tensor(void* data, DType dtype, std::initializer_list<int64_t> dims)
    : data(data), dtype(dtype), ndim(int(dims.size())) {
    if (dims.size() > kMaxDims) throw std::invalid_argument("Tensor supports at most 4 dims");
    std::copy(dims.begin(), dims.end(), shape.begin());
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
}

}  // namespace engine
