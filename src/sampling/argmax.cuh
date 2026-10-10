#pragma once

#include <climits>
#include <cmath>

namespace engine {

constexpr int kArgmaxThreads = 1024;

__device__ inline bool better(float v, int i, float best, int arg) {
    return i != INT_MAX && (arg == INT_MAX || v > best || (v == best && i < arg));
}

template <typename Score>
__device__ int block_argmax(int vocab, Score score) {
    __shared__ float values[kArgmaxThreads];
    __shared__ int args[kArgmaxThreads];

    float best = -INFINITY;
    int arg = INT_MAX;
    for (int i = threadIdx.x; i < vocab; i += kArgmaxThreads) {
        const float v = score(i);
        if (better(v, i, best, arg)) {
            best = v;
            arg = i;
        }
    }
    values[threadIdx.x] = best;
    args[threadIdx.x] = arg;
    __syncthreads();

    for (int stride = kArgmaxThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const int other = threadIdx.x + stride;
            if (better(values[other], args[other], values[threadIdx.x], args[threadIdx.x])) {
                values[threadIdx.x] = values[other];
                args[threadIdx.x] = args[other];
            }
        }
        __syncthreads();
    }
    return args[0];
}

}  // namespace engine
