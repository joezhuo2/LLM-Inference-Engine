## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 6939 us, median busy 6884 us, median idle 0.8% of the span, 0 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 12338 us, median busy 10214 us, median idle 19.5% of the span, 54 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| forward_decode | 224.0 | 7176.7 | 100.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1676.6 | 74.7 | 90.1 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5632 | 99.6 | 17.7 | 5.4 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 25.1 | 142.6 | 1.3 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 18.1 | 1.6 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 14.6 | 2.6 | 0.8 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.9 | 1.4 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.3 | 1.0 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.2 | 48.3 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 4.1 | 46.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.8 | 10.9 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.7 | 3.3 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.2 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 8.23 ms
- weights read per token / TPOT: 251.4 GB/s = 77.7% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.88 ms (84% of TPOT)
- weights / kernel busy time: 300.5 GB/s (what the kernels would reach with no launch gaps)

