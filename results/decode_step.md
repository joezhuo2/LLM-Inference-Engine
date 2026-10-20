## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 9343 us, median busy 7002 us, median idle 24.3% of the span, 39 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 13114 us, median busy 9899 us, median idle 24.4% of the span, 66 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3093.3 | 43.6 |
| down | 22.0 | 1705.0 | 24.0 |
| qkv | 22.0 | 743.3 | 10.5 |
| o_proj | 22.0 | 606.0 | 8.5 |
| lm_head | 1.0 | 394.5 | 5.6 |
| attention | 22.0 | 271.7 | 3.8 |
| add1_norm2 | 22.0 | 92.4 | 1.3 |
| add2_norm1 | 21.0 | 84.7 | 1.2 |
| rope | 22.0 | 49.6 | 0.7 |
| write_kv | 22.0 | 27.4 | 0.4 |
| swiglu | 22.0 | 20.3 | 0.3 |
| embedding | 1.0 | 2.8 | 0.0 |
| norm1 | 1.0 | 2.7 | 0.0 |
| final_norm | 1.0 | 2.7 | 0.0 |
| add2 | 1.0 | 0.8 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1650.2 | 73.6 | 90.1 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 77.8 | 13.8 | 4.3 |
| engine::fast::add_rmsnorm_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const __nv... | 11008 | 45.7 | 4.2 | 2.5 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 20.9 | 118.9 | 1.1 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 12.7 | 2.3 | 0.7 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.0 | 1.3 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.1 | 0.9 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.1 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.0 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.2 | 8.7 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.4 | 2.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.7 | 2.9 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 11.01 ms
- weights read per token / TPOT: 187.9 GB/s = 58.1% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.00 ms (64% of TPOT)
- weights / kernel busy time: 295.5 GB/s (what the kernels would reach with no launch gaps)

