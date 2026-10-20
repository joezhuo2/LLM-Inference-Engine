#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/kernels/paged_attention.h"
#include "engine/kernels/paged_attention_v2.h"
using namespace engine;
int main(int argc, char** argv) {
    const int heads = 32, kvh = 4, hd = 64, bs = 16;
    int cfgs[][2] = {{1, 128},  {1, 512},  {1, 2048},  {1, 8192},
                     {16, 128}, {16, 512}, {16, 2048}, {16, 8192}};
    int nc = sizeof(cfgs) / sizeof(cfgs[0]);
    int only[1][2] = {{0, 0}};
    if (argc == 3) {
        only[0][0] = atoi(argv[1]);
        only[0][1] = atoi(argv[2]);
    }
    for (int ci = 0; ci < (argc == 3 ? 1 : nc); ++ci) {
        auto& c = argc == 3 ? only[0] : cfgs[ci];
        int rows = c[0], ctx = c[1];
        int blocks = ctx / bs;
        int nblocks = rows * blocks;
        size_t cache_n = (size_t)nblocks * kvh * bs * hd;
        size_t qkv_n = (size_t)rows * (heads + 2 * kvh) * hd;
        void *k, *v, *qkv, *out, *bt, *cl;
        cudaMalloc(&k, cache_n * 2);
        cudaMalloc(&v, cache_n * 2);
        cudaMalloc(&qkv, qkv_n * 2);
        cudaMalloc(&out, (size_t)rows * heads * hd * 2);
        std::vector<unsigned short> h(cache_n);
        for (auto& x : h) x = 0x3c00 + (rand() & 0x3ff) - 0x200;
        cudaMemcpy(k, h.data(), cache_n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(v, h.data(), cache_n * 2, cudaMemcpyHostToDevice);
        std::vector<unsigned short> hq(qkv_n);
        for (auto& x : hq) x = 0x3c00 + (rand() & 0x3ff) - 0x200;
        cudaMemcpy(qkv, hq.data(), qkv_n * 2, cudaMemcpyHostToDevice);
        std::vector<int> t(rows * blocks), l(rows, ctx);
        for (int i = 0; i < rows * blocks; ++i) t[i] = i;
        cudaMalloc(&bt, t.size() * 4);
        cudaMalloc(&cl, l.size() * 4);
        cudaMemcpy(bt, t.data(), t.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(cl, l.data(), l.size() * 4, cudaMemcpyHostToDevice);
        Tensor To(out, DType::BF16, {rows, heads * hd}),
            Tq(qkv, DType::BF16, {rows, (heads + 2 * kvh) * hd});
        Tensor Tk(k, DType::BF16, {nblocks, kvh, bs, hd}),
            Tv(v, DType::BF16, {nblocks, kvh, bs, hd});
        Tensor Tb(bt, DType::I32, {rows, blocks}), Tc(cl, DType::I32, {rows});
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
        float v2 = timeit([&] { paged_attention_decode_v2(To, Tq, Tk, Tv, Tb, Tc, nullptr); });
        float v1 = -1;
        try {
            v1 = timeit([&] { paged_attention_decode(To, Tq, Tk, Tv, Tb, Tc, nullptr); });
        } catch (...) {
        }
        printf("rows %2d ctx %5d: v1 %8.1f us  v2 %8.1f us\n", rows, ctx, v1, v2);
        cudaFree(k);
        cudaFree(v);
        cudaFree(qkv);
        cudaFree(out);
        cudaFree(bt);
        cudaFree(cl);
    }
}
