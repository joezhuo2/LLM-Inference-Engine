## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 8198 us, median busy 6918 us, median idle 11.8% of the span, 15 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 11352 us, median busy 10152 us, median idle 9.9% of the span, 46 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3161.5 | 43.9 |
| down | 22.0 | 1743.4 | 24.2 |
| qkv | 22.0 | 764.1 | 10.6 |
| o_proj | 22.0 | 621.8 | 8.6 |
| lm_head | 1.0 | 402.7 | 5.6 |
| attention | 22.0 | 317.1 | 4.4 |
| rope | 22.0 | 59.5 | 0.8 |
| add1_norm2 | 22.0 | 37.1 | 0.5 |
| add2_norm1 | 21.0 | 33.7 | 0.5 |
| write_kv | 22.0 | 30.9 | 0.4 |
| swiglu | 22.0 | 24.5 | 0.3 |
| embedding | 1.0 | 3.2 | 0.0 |
| norm1 | 1.0 | 3.2 | 0.0 |
| final_norm | 1.0 | 3.1 | 0.0 |
| add2 | 1.0 | 0.9 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1688.4 | 75.3 | 90.8 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 89.8 | 16.0 | 4.8 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.0 | 119.1 | 1.1 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 18.2 | 1.7 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 15.3 | 2.7 | 0.8 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.9 | 1.4 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 6.2 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.4 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.6 | 10.2 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.6 | 3.2 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.2 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 9.28 ms
- weights read per token / TPOT: 222.9 GB/s = 68.9% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.92 ms (75% of TPOT)
- weights / kernel busy time: 299.1 GB/s (what the kernels would reach with no launch gaps)

