#pragma once

#include <vector>

#include "engine/runtime/device_buffer.h"
#include "kernels/cuda_check.h"

namespace engine::test {

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() { CUDA_CHECK(cudaStreamDestroy(stream_)); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    operator cudaStream_t() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
DeviceBuffer upload(const std::vector<T>& host) {
    DeviceBuffer d(host.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(d.data(), host.data(), d.size(), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& d) {
    std::vector<T> host(d.size() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(host.data(), d.data(), d.size(), cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace engine::test
