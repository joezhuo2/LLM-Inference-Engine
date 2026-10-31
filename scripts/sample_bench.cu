#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <random>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/sampling/greedy.h"
#include "engine/sampling/sample.h"

using namespace engine;

__global__ void scale_kernel(float* keys, int* idx, const __nv_bfloat16* logits, int n, float t) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = __bfloat162float(logits[i]) / t;
        idx[i] = i;
    }
}

int main(int argc, char** argv) {
    const int vocab = argc > 2 ? atoi(argv[2]) : 32000;
    std::vector<int> row_counts = {1, 16};
    if (argc > 1) row_counts = {atoi(argv[1])};
    std::mt19937 gen(0);
    std::normal_distribution<float> normal(0.0f, 2.5f);
    for (int rows : row_counts) {
        const size_t n = size_t(rows) * vocab;
        std::vector<__nv_bfloat16> h(n);
        for (auto& x : h) x = __float2bfloat16(normal(gen));
        void *logits, *ids, *row_params, *keys_in, *keys_out, *idx_in, *idx_out, *offsets;
        cudaMalloc(&logits, n * 2);
        cudaMalloc(&ids, size_t(rows) * 4);
        cudaMalloc(&row_params, size_t(rows) * sizeof(SampleRow));
        cudaMalloc(&keys_in, n * 4);
        cudaMalloc(&keys_out, n * 4);
        cudaMalloc(&idx_in, n * 4);
        cudaMalloc(&idx_out, n * 4);
        cudaMalloc(&offsets, size_t(rows + 1) * 4);
        cudaMemcpy(logits, h.data(), n * 2, cudaMemcpyHostToDevice);
        std::vector<int> off(rows + 1);
        for (int r = 0; r <= rows; ++r) off[r] = r * vocab;
        cudaMemcpy(offsets, off.data(), off.size() * 4, cudaMemcpyHostToDevice);
        const Tensor L(logits, DType::BF16, {rows, vocab}), I(ids, DType::I32, {rows});
        auto timeit = [&](auto fn) {
            for (int i = 0; i < 3; ++i) fn();
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            const int N = 50;
            for (int i = 0; i < N; ++i) fn();
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            return ms * 1000 / N;
        };
        auto sampled = [&](float t, int k, float p) {
            std::vector<SampleRow> r(rows);
            for (int i = 0; i < rows; ++i) r[i] = {{t, uint64_t(i), k, p}, 7};
            cudaMemcpy(row_params, r.data(), r.size() * sizeof(SampleRow), cudaMemcpyHostToDevice);
            return timeit(
                [&] { sample(I, L, static_cast<const SampleRow*>(row_params), nullptr); });
        };
        const float us_greedy = timeit([&] { greedy(I, L, nullptr); });
        const float us_temp = sampled(0.7f, 0, 1.0f);
        const float us_k = sampled(0.7f, 50, 1.0f);
        const float us_p = sampled(0.7f, 0, 0.9f);
        const float us_kp = sampled(0.7f, 50, 0.9f);
        size_t temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, temp_bytes, static_cast<float*>(keys_in), static_cast<float*>(keys_out),
            static_cast<int*>(idx_in), static_cast<int*>(idx_out), int(n), rows,
            static_cast<int*>(offsets), static_cast<int*>(offsets) + 1);
        void* temp;
        cudaMalloc(&temp, temp_bytes);
        const float us_sort = timeit([&] {
            scale_kernel<<<unsigned((n + 255) / 256), 256>>>(
                static_cast<float*>(keys_in), static_cast<int*>(idx_in),
                static_cast<const __nv_bfloat16*>(logits), int(n), 0.7f);
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                temp, temp_bytes, static_cast<float*>(keys_in), static_cast<float*>(keys_out),
                static_cast<int*>(idx_in), static_cast<int*>(idx_out), int(n), rows,
                static_cast<int*>(offsets), static_cast<int*>(offsets) + 1);
        });
        printf(
            "rows %2d vocab %d: greedy %7.1f  temperature %7.1f  top-k 50 %7.1f  top-p 0.9 %7.1f  "
            "top-k + top-p %7.1f  full sort %7.1f us\n",
            rows, vocab, us_greedy, us_temp, us_k, us_p, us_kp, us_sort);
        for (void* p : {logits, ids, row_params, keys_in, keys_out, idx_in, idx_out, offsets, temp})
            cudaFree(p);
    }
}
