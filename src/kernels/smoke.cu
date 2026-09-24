#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_check.h"
#include "engine/kernels/smoke.h"

namespace engine::kernels {

namespace {

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

}  // namespace

float smoke_vector_add(int n) {
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = 0.5f * i;
        b[i] = 1.0f - i;
    }
    const size_t bytes = size_t(n) * sizeof(float);
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    float err = 0.0f;
    for (int i = 0; i < n; ++i) err = std::max(err, std::fabs(c[i] - (a[i] + b[i])));
    return err;
}

}  // namespace engine::kernels
