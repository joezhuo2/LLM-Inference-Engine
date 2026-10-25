#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {2048, 4096, 8192};
    const int iters = 20;
    cublasHandle_t handle;
    cublasCreate(&handle);
    double best = 0.0;
    for (const int n : sizes) {
        const size_t elems = size_t(n) * size_t(n);
        std::vector<unsigned short> h(elems);
        for (auto& v : h) v = 0x3c00 + (rand() & 0xff) - 0x80;
        void *a, *b, *c;
        cudaMalloc(&a, elems * 2);
        cudaMalloc(&b, elems * 2);
        cudaMalloc(&c, elems * 2);
        cudaMemcpy(a, h.data(), elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(b, h.data(), elems * 2, cudaMemcpyHostToDevice);
        const float alpha = 1.0f, beta = 0.0f;
        auto run = [&] {
            return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, a, CUDA_R_16BF,
                                n, b, CUDA_R_16BF, n, &beta, c, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
        };
        for (int i = 0; i < 3; ++i) run();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) run();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess || run() != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "gemm %d failed: %s\n", n, cudaGetErrorString(err));
            return 1;
        }
        const double us = 1e3 * ms / iters;
        const double tflops = 2.0 * n * double(n) * n / (us * 1e-6) / 1e12;
        printf("bf16 gemm %d x %d x %d: %.1f us, %.1f TFLOP/s\n", n, n, n, us, tflops);
        if (tflops > best) best = tflops;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }
    cublasDestroy(handle);
    printf("peak_tflops=%.1f\n", best);
}
