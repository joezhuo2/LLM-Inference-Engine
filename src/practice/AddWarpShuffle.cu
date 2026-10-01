#include <cuda_runtime.h>

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sumWarpShuffle(const float* input, float* result, int num_elements) {
    float sum = 0.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        sum = input[idx];
    }

    sum = warpReduceSum(sum);

    static __shared__ float shared[32];

    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (lane == 0) {
        shared[warpId] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (warpId == 0) {
        sum = warpReduceSum(sum);
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}