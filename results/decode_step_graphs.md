## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 7640 us, median busy 7587 us, median idle 0.7% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 13836 us, median busy 11380 us, median idle 15.7% of the span, 54 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 224.0 | 7765.9 | 100.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1644.6 | 73.3 | 82.0 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 249.5 | 44.3 | 12.4 |
| engine::fast::add_rmsnorm_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const __nv... | 11008 | 52.2 | 4.7 | 2.6 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.1 | 119.9 | 1.1 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 13.9 | 2.5 | 0.7 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.5 | 1.3 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.2 | 0.9 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.5 | 39.8 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 38.0 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.6 | 10.2 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.6 | 3.1 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 8.37 ms
- weights read per token / TPOT: 247.1 GB/s = 76.4% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.59 ms (91% of TPOT)
- weights / kernel busy time: 272.7 GB/s (what the kernels would reach with no launch gaps)

