## Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 8782 us, median busy 7725 us, median idle 11.4% of the span, 18 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 13411 us, median busy 11564 us, median idle 12.4% of the span, 59 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3118.8 | 40.0 |
| down | 22.0 | 1715.8 | 22.0 |
| attention | 22.0 | 880.4 | 11.3 |
| qkv | 22.0 | 750.6 | 9.6 |
| o_proj | 22.0 | 612.7 | 7.8 |
| lm_head | 1.0 | 395.6 | 5.1 |
| rope | 22.0 | 96.7 | 1.2 |
| norm2 | 22.0 | 71.6 | 0.9 |
| norm1 | 22.0 | 67.4 | 0.9 |
| write_kv | 22.0 | 29.7 | 0.4 |
| swiglu | 22.0 | 22.8 | 0.3 |
| add1 | 22.0 | 19.3 | 0.2 |
| add2 | 22.0 | 18.7 | 0.2 |
| embedding | 1.0 | 3.1 | 0.0 |
| final_norm | 1.0 | 3.0 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1663.2 | 74.1 | 82.5 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 236.2 | 41.9 | 11.7 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 36.6 | 3.2 | 1.8 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 24.9 | 4.4 | 1.2 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 21.0 | 119.1 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 10.0 | 0.9 | 0.5 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.6 | 1.4 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.4 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.1 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.3 | 37.2 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.5 | 9.7 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 9.53 ms
- weights read per token / TPOT: 217.1 GB/s = 67.1% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.73 ms (81% of TPOT)
- weights / kernel busy time: 267.8 GB/s (what the kernels would reach with no launch gaps)

