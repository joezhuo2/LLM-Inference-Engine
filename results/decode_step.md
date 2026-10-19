## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 8650 us, median busy 7635 us, median idle 10.0% of the span, 15 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 12556 us, median busy 11261 us, median idle 9.1% of the span, 47 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3073.5 | 39.9 |
| down | 22.0 | 1692.7 | 22.0 |
| attention | 22.0 | 882.9 | 11.5 |
| qkv | 22.0 | 740.6 | 9.6 |
| o_proj | 22.0 | 605.2 | 7.9 |
| lm_head | 1.0 | 390.2 | 5.1 |
| add1_norm2 | 22.0 | 106.3 | 1.4 |
| add2_norm1 | 21.0 | 94.7 | 1.2 |
| rope | 22.0 | 56.1 | 0.7 |
| write_kv | 22.0 | 30.4 | 0.4 |
| swiglu | 22.0 | 23.2 | 0.3 |
| embedding | 1.0 | 3.1 | 0.0 |
| norm1 | 1.0 | 3.0 | 0.0 |
| final_norm | 1.0 | 3.0 | 0.0 |
| add2 | 1.0 | 0.9 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1640.1 | 73.1 | 82.4 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 236.8 | 42.1 | 11.9 |
| engine::fast::add_rmsnorm_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const __nv... | 11008 | 51.8 | 4.7 | 2.6 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 20.7 | 117.4 | 1.0 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 14.4 | 2.6 | 0.7 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.8 | 1.4 | 0.4 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 5.8 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.9 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.2 | 36.6 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.5 | 9.7 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.6 | 3.0 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 9.30 ms
- weights read per token / TPOT: 222.6 GB/s = 68.8% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.64 ms (82% of TPOT)
- weights / kernel busy time: 271.0 GB/s (what the kernels would reach with no launch gaps)

