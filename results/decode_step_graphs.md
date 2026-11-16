## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 6434 us, median busy 6346 us, median idle 1.4% of the span, 3 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 8532 us, median busy 7258 us, median idle 15.0% of the span, 36 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 224.0 | 6385.6 | 100.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1522.7 | 67.9 | 92.8 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5544 | 56.8 | 10.3 | 3.5 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 19.3 | 109.9 | 1.2 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 11.8 | 1.1 | 0.7 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 9.7 | 1.7 | 0.6 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 5.2 | 0.9 | 0.3 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 3.6 | 0.7 | 0.2 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.0 | 34.2 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.0 | 34.1 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 1.9 | 7.2 | 0.1 |
| void engine::<unnamed>::flash_attention_prefill_kernel<(int)64>(__nv_bfloat16... | 88 | 1.2 | 13.6 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.1 | 2.2 | 0.1 |

## Decode bandwidth

- max_batch 1, median TPOT 6.89 ms
- weights read per token / TPOT: 300.1 GB/s = 92.7% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.35 ms (92% of TPOT)
- weights / kernel busy time: 326.0 GB/s (what the kernels would reach with no launch gaps)

