#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "AddNaiveAtomics.cu"
#include "AddSharedMemory.cu"
#include "AddWarpShuffle.cu"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " "                    \
                      << cudaGetErrorString(err) << std::endl;                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

template <typename Launch>
void bench(const char* name, Launch launch, float* d_result) {
    constexpr int runs = 5;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(runs);
    for (float& t : times) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    }

    float sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    std::sort(times.begin(), times.end());
    std::cout << "[" << name << "] Median: " << times[runs / 2] << " ms (min "
              << times.front() << ", max " << times.back() << ") | Sum: " << sum
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    constexpr int num_elements = 16777216;
    constexpr size_t bytes = num_elements * sizeof(float);
    constexpr int block_size = 256;
    constexpr int grid_size = (num_elements + block_size - 1) / block_size;

    std::vector<float> h_input(num_elements, 1.0f);

    float *d_input, *d_result;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    bench("Naive Atomics", [&] {
        sumNaiveAtomics<<<grid_size, block_size>>>(d_input, d_result, num_elements);
    }, d_result);

    bench("Shared Memory", [&] {
        sumSharedMemory<<<grid_size, block_size, block_size * sizeof(float)>>>(
            d_input, d_result, num_elements);
    }, d_result);

    bench("Warp Shuffle", [&] {
        sumWarpShuffle<<<grid_size, block_size, block_size * sizeof(float)>>>(
            d_input, d_result, num_elements);
    }, d_result);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}