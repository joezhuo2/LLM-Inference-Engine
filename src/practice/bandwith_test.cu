#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

float* upload(const std::vector<float>& h) {
    size_t bytes = h.size() * sizeof(float);
    float* d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice);
    return d;
}

void launch(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(a, b, c, n);
}

float timeKernel(const float* a, const float* b, float* c, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

bool verify(const float* d, int n, float expected) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        if (h[i] != expected) {
            std::cout << "Mismatch at " << i << ": " << h[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = n * sizeof(float);

    float* a = upload(std::vector<float>(n, 1.0f));
    float* b = upload(std::vector<float>(n, 2.0f));
    float* c;
    cudaMalloc(&c, bytes);

    launch(a, b, c, n);
    cudaDeviceSynchronize();
    cudaMemset(c, 0, bytes);

    float ms = timeKernel(a, b, c, n);
    double gbps = (3.0 * bytes) / (ms / 1000.0) / 1e9;
    std::cout << "Bandwidth: " << gbps << " GB/s" << std::endl;

    if (verify(c, n, 3.0f)) {
        std::cout << "Correct" << std::endl;
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}