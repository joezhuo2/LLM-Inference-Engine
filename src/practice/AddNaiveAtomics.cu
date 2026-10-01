#include <cuda_runtime.h>

__global__ void sumNaiveAtomics(const float* input, float* result, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        atomicAdd(result, input[idx]);
    }
}