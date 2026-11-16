## Launch gaps per forward step

- decode: 63 steps, 224 kernels, median span 9092 us, median busy 7897 us, median idle 13.3% of the span, 19 gaps over 5 us per step
- prefill: 1 steps, 224 kernels, median span 68484 us, median busy 66546 us, median idle 2.8% of the span, 90 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 2987.3 | 38.1 |
| down | 22.0 | 1598.8 | 20.4 |
| attention | 22.0 | 1408.9 | 17.9 |
| qkv | 22.0 | 725.4 | 9.2 |
| o_proj | 22.0 | 576.4 | 7.3 |
| lm_head | 1.0 | 377.7 | 4.8 |
| rope | 22.0 | 41.2 | 0.5 |
| swiglu | 22.0 | 33.2 | 0.4 |
| add2_norm1 | 21.0 | 33.0 | 0.4 |
| add1_norm2 | 22.0 | 31.7 | 0.4 |
| write_kv | 22.0 | 25.6 | 0.3 |
| norm1 | 1.0 | 4.0 | 0.1 |
| final_norm | 1.0 | 3.9 | 0.1 |
| embedding | 1.0 | 1.1 | 0.0 |
| add2 | 1.0 | 1.1 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_12... | 2836 | 258.1 | 91.0 | 45.9 |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_12... | 2772 | 137.0 | 49.4 | 24.4 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 1386 | 88.8 | 64.0 | 15.8 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 88 | 61.4 | 697.7 | 10.9 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 2752 | 4.9 | 1.8 | 0.9 |
| engine::fast::swiglu_vec_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long,... | 1408 | 3.4 | 2.4 | 0.6 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 1408 | 2.9 | 2.1 | 0.5 |
| void engine::<unnamed>::flash_attention_prefill_kernel<(int)64>(__nv_bfloat16... | 22 | 2.0 | 92.8 | 0.4 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 1408 | 1.8 | 1.3 | 0.3 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 64 | 0.7 | 11.3 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 128 | 0.5 | 4.1 | 0.1 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 64 | 0.1 | 1.7 | 0.0 |

## Decode bandwidth

- max_batch 16, median TPOT 9.78 ms
- weights read per token / TPOT: 211.5 GB/s = 65.3% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.90 ms (81% of TPOT)
- weights / kernel busy time: 262.0 GB/s (what the kernels would reach with no launch gaps)

