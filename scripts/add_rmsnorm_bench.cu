#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/kernels/add.h"
#include "engine/kernels/add_rmsnorm.h"
#include "engine/kernels/rmsnorm.h"

using namespace engine;

int main(int argc, char** argv) {
    const int64_t tokens = argc > 1 ? atoi(argv[1]) : 1;
    const int64_t hidden = argc > 2 ? atoi(argv[2]) : 2048;
    const size_t n = size_t(tokens * hidden);
    std::vector<unsigned short> h(n);
    for (auto& v : h) v = 0x3c00 + (rand() & 0x3ff) - 0x200;
    void *x, *r, *s, *o, *w;
    cudaMalloc(&x, n * 2);
    cudaMalloc(&r, n * 2);
    cudaMalloc(&s, n * 2);
    cudaMalloc(&o, n * 2);
    cudaMalloc(&w, size_t(hidden) * 2);
    cudaMemcpy(x, h.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(r, h.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, h.data(), size_t(hidden) * 2, cudaMemcpyHostToDevice);
    const Tensor X(x, DType::BF16, {tokens, hidden}), R(r, DType::BF16, {tokens, hidden}),
        S(s, DType::BF16, {tokens, hidden}), O(o, DType::BF16, {tokens, hidden}),
        W(w, DType::BF16, {hidden});
    for (int i = 0; i < 6; ++i) {
        naive::add(S, X, R, nullptr);
        naive::rmsnorm(O, S, W, 1e-5f, nullptr);
        fast::add_rmsnorm(S, O, X, R, W, 1e-5f, nullptr);
    }
    cudaDeviceSynchronize();
    printf("done tokens %ld hidden %ld\n", long(tokens), long(hidden));
}
