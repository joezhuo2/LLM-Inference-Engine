## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 8069 us, median busy 6909 us, median idle 11.5% of the span, 17 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 11187 us, median busy 10093 us, median idle 8.1% of the span, 40 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3138.8 | 43.7 |
| down | 22.0 | 1724.2 | 24.0 |
| qkv | 22.0 | 759.2 | 10.6 |
| o_proj | 22.0 | 619.1 | 8.6 |
| lm_head | 1.0 | 398.3 | 5.6 |
| attention | 22.0 | 332.8 | 4.6 |
| rope | 22.0 | 61.7 | 0.9 |
| add1_norm2 | 22.0 | 38.5 | 0.5 |
| add2_norm1 | 21.0 | 34.9 | 0.5 |
| write_kv | 22.0 | 32.0 | 0.4 |
| swiglu | 22.0 | 25.2 | 0.4 |
| norm1 | 1.0 | 3.3 | 0.0 |
| final_norm | 1.0 | 3.3 | 0.0 |
| embedding | 1.0 | 3.3 | 0.0 |
| add2 | 1.0 | 0.9 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1674.8 | 74.7 | 90.4 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 94.5 | 16.8 | 5.1 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.2 | 120.4 | 1.1 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 18.9 | 1.7 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 15.8 | 2.8 | 0.9 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 8.2 | 1.5 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 6.3 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.5 | 39.5 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.1 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.7 | 10.6 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.7 | 3.3 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.3 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 8.85 ms
- weights read per token / TPOT: 233.7 GB/s = 72.2% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.91 ms (78% of TPOT)
- weights / kernel busy time: 299.4 GB/s (what the kernels would reach with no launch gaps)

