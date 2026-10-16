## Launch gaps per forward step

- decode: 252 steps, 267 kernels, median span 8703 us, median busy 7648 us, median idle 11.2% of the span, 17 gaps over 5 us per step
- prefill: 4 steps, 267 kernels, median span 12997 us, median busy 11685 us, median idle 8.3% of the span, 50 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3057.6 | 39.9 |
| down | 22.0 | 1686.4 | 22.0 |
| attention | 22.0 | 868.5 | 11.3 |
| qkv | 22.0 | 737.5 | 9.6 |
| o_proj | 22.0 | 602.3 | 7.9 |
| lm_head | 1.0 | 388.2 | 5.1 |
| rope | 22.0 | 96.7 | 1.3 |
| norm2 | 22.0 | 71.1 | 0.9 |
| norm1 | 22.0 | 66.9 | 0.9 |
| write_kv | 22.0 | 29.4 | 0.4 |
| swiglu | 22.0 | 22.7 | 0.3 |
| add1 | 22.0 | 19.3 | 0.3 |
| add2 | 22.0 | 18.7 | 0.2 |
| embedding | 1.0 | 3.0 | 0.0 |
| final_norm | 1.0 | 3.0 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1632.6 | 72.8 | 82.4 |
| engine::paged_attention_decode_kernel(__nv_bfloat16 *, const __nv_bfloat16 *,... | 5632 | 233.6 | 41.5 | 11.8 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 11520 | 36.4 | 3.2 | 1.8 |
| engine::naive::<unnamed>::rope_kernel(__nv_bfloat16 *, const int *, int, int,... | 5632 | 24.9 | 4.4 | 1.3 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 20.6 | 117.2 | 1.0 |
| engine::naive::<unnamed>::add_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, ... | 11264 | 10.0 | 0.9 | 0.5 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 7.5 | 1.3 | 0.4 |
| engine::naive::<unnamed>::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 ... | 5632 | 6.4 | 1.1 | 0.3 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.4 | 38.4 | 0.2 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.2 | 36.9 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 2.5 | 9.6 | 0.1 |
| engine::naive::<unnamed>::embedding_kernel(__nv_bfloat16 *, const int *, cons... | 256 | 0.8 | 3.1 | 0.0 |

## Decode bandwidth

- max_batch 1, median TPOT 9.37 ms
- weights read per token / TPOT: 220.9 GB/s = 68.3% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 7.65 ms (82% of TPOT)
- weights / kernel busy time: 270.5 GB/s (what the kernels would reach with no launch gaps)

