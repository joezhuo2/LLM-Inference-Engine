## Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 7923 us, median busy 7854 us, median idle 0.9% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 12683 us, median busy 11595 us, median idle 8.5% of the span, 48 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 267.0 | 8049.3 | 100.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1702.0 | 75.9 | 81.8 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 258.7 | 45.9 | 12.4 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 38.7 | 3.4 | 1.9 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 25.5 | 4.5 | 1.2 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.7 | 123.5 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 9.2 | 0.8 | 0.4 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.8 | 1.4 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.1 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.6 | 41.2 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.5 | 39.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.7 | 10.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.2 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 8.55 ms
- weights read per token / TPOT: 242.0 GB/s = 74.8% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.85 ms (92% of TPOT)
- weights / kernel busy time: 263.4 GB/s (what the kernels would reach with no launch gaps)

