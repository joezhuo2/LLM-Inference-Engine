## Launch gaps per forward step

- decode: 63 steps, 224 kernels, median span 9731 us, median busy 8543 us, median idle 11.6% of the span, 17 gaps over 5 us per step
- prefill: 1 steps, 224 kernels, median span 112978 us, median busy 111443 us, median idle 1.4% of the span, 106 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3172.5 | 37.2 |
| down | 22.0 | 1715.6 | 20.1 |
| attention | 22.0 | 1631.2 | 19.1 |
| qkv | 22.0 | 775.1 | 9.1 |
| o_proj | 22.0 | 623.6 | 7.3 |
| lm_head | 1.0 | 400.1 | 4.7 |
| rope | 22.0 | 50.8 | 0.6 |
| swiglu | 22.0 | 38.8 | 0.5 |
| add2_norm1 | 21.0 | 38.0 | 0.4 |
| add1_norm2 | 22.0 | 37.4 | 0.4 |
| write_kv | 22.0 | 30.0 | 0.4 |
| norm1 | 1.0 | 4.5 | 0.1 |
| final_norm | 1.0 | 4.5 | 0.1 |
| embedding | 1.0 | 4.2 | 0.0 |
| add2 | 1.0 | 1.2 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_12... | 2836 | 274.3 | 96.7 | 42.2 |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_12... | 2772 | 147.4 | 53.2 | 22.7 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 1408 | 139.2 | 98.9 | 21.4 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 88 | 71.6 | 813.4 | 11.0 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 2752 | 5.7 | 2.1 | 0.9 |
| engine::fast::swiglu_vec_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long,... | 1408 | 3.8 | 2.7 | 0.6 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 1408 | 3.6 | 2.6 | 0.6 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 1408 | 2.1 | 1.5 | 0.3 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 64 | 0.8 | 12.9 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 128 | 0.6 | 4.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 64 | 0.3 | 4.6 | 0.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 64 | 0.1 | 1.9 | 0.0 |

## Decode bandwidth

- max_batch 16, median TPOT 10.60 ms
- weights read per token / TPOT: 195.1 GB/s = 60.3% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 8.54 ms (81% of TPOT)
- weights / kernel busy time: 242.2 GB/s (what the kernels would reach with no launch gaps)

