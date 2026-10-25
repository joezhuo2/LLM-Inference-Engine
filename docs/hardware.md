NVIDIA GeForce RTX 5060 Laptop GPU
compute capability: 12.0
SMs: 26
VRAM: 8.52 GB
L2: 32768 KB
shared mem per SM: 100 KB

Measured ceilings for the roofline (2026-10-25):
- DRAM bandwidth: 323.6 GB/s (`src/practice/bandwith_test.cu`)
- BF16 tensor core GEMM: 34.5 TFLOP/s, the best of square cuBLAS GEMMs at 2048, 4096 and 8192 with F32 accumulation (`nvcc -std=c++20 -arch=sm_120 scripts/gemm_peak.cu -lcublas -o /tmp/gemm_peak && /tmp/gemm_peak`; 4096 and 8192 give 31.5 and 32.0)
- ridge point: 34.5 TFLOP/s / 323.6 GB/s = 107 FLOP/byte
