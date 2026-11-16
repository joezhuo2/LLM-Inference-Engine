## Launch gaps per forward step

- decode: 252 steps, 224 kernels, median span 7556 us, median busy 6392 us, median idle 14.9% of the span, 20 gaps over 5 us per step
- prefill: 4 steps, 224 kernels, median span 9261 us, median busy 7095 us, median idle 22.7% of the span, 49 gaps over 5 us per step

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 2874.5 | 44.8 |
| down | 22.0 | 1577.4 | 24.6 |
| qkv | 22.0 | 687.0 | 10.7 |
| o_proj | 22.0 | 556.7 | 8.7 |
| lm_head | 1.0 | 367.2 | 5.7 |
| attention | 22.0 | 219.1 | 3.4 |
| rope | 22.0 | 40.4 | 0.6 |
| add1_norm2 | 22.0 | 25.1 | 0.4 |
| add2_norm1 | 21.0 | 23.7 | 0.4 |
| write_kv | 22.0 | 22.1 | 0.3 |
| swiglu | 22.0 | 16.8 | 0.3 |
| final_norm | 1.0 | 2.2 | 0.0 |
| norm1 | 1.0 | 2.2 | 0.0 |
| embedding | 1.0 | 0.9 | 0.0 |
| add2 | 1.0 | 0.6 | 0.0 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat... | 22432 | 1529.3 | 68.2 | 92.8 |
| engine::paged_attention_decode_v2_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 5544 | 55.2 | 10.0 | 3.4 |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x... | 176 | 19.3 | 109.4 | 1.2 |
| engine::fast::add_rmsnorm_vec_kernel(__nv_bfloat16 *, __nv_bfloat16 *, const ... | 11008 | 12.6 | 1.1 | 0.8 |
| engine::fast::rope_kernel(__nv_bfloat16 *, const int *, int, int, int, engine... | 5632 | 10.4 | 1.8 | 0.6 |
| engine::write_kv_kernel(const __nv_bfloat16 *, const int *, __nv_bfloat16 *, ... | 5632 | 5.7 | 1.0 | 0.3 |
| engine::fast::swiglu_kernel(__nv_bfloat16 *, const __nv_bfloat16 *, long, long) | 5544 | 4.2 | 0.8 | 0.3 |
| nvjet_sm120_tst_mma_112x64x64_4_112x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 3.0 | 34.1 | 0.2 |
| nvjet_sm120_tst_mma_80x64x64_5_80x8x64_tmaAB_alignCD4_bz_TNNN | 88 | 2.9 | 33.4 | 0.2 |
| engine::<unnamed>::sample_kernel(int *, const __nv_bfloat16 *, int, const eng... | 256 | 1.8 | 7.1 | 0.1 |
| engine::naive::<unnamed>::rmsnorm_kernel(__nv_bfloat16 *, const __nv_bfloat16... | 512 | 1.1 | 2.2 | 0.1 |
| void engine::<unnamed>::flash_attention_prefill_kernel<(int)64>(__nv_bfloat16... | 88 | 1.1 | 12.8 | 0.1 |

## Decode bandwidth

- max_batch 1, median TPOT 8.65 ms
- weights read per token / TPOT: 239.2 GB/s = 73.9% of the 323.6 GB/s peak (target 60%)
- GPU kernel busy time in a decode step: 6.39 ms (74% of TPOT)
- weights / kernel busy time: 323.7 GB/s (what the kernels would reach with no launch gaps)

