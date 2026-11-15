# Sharded forward pass, runs of scripts/sharded_forward_bench.cu

TinyLlama, RTX 5060 Laptop GPU, release build, commit 284c886, 2026-11-14. Each row runs in its own process. Three prompts (the start of a story, a question, a Python function to finish), each with BOS, and 64 decode steps after each; ms per decode step is the median and min to max over the 189 timed steps (the first step of each prompt is dropped). The largest logit difference is over every prefill row and every decode step of the three prompts, both models fed the tokens of the whole model; greedy tokens equal counts the tokens of 64-token greedy answers that match the whole model before the first difference.

### Run 1

| layout | MiB on the device | largest logit difference | greedy tokens equal | ms per decode step | min to max |
| --- | --- | --- | --- | --- | --- |
| whole model, ModelRunner | 2222.0 | reference | reference | 8.57 | 7.93 to 12.38 |
| 1 rank | 2222.0 | 0.0000 | 192 of 192 | 8.60 | 7.79 to 22.43 |
| 2 ranks | 2474.0 | 0.7500 | 192 of 192 | 11.21 | 10.46 to 20.85 |
| 4 ranks | 2978.0 | 0.6406 | 192 of 192 | 17.89 | 15.79 to 38.99 |

### Run 2

| layout | MiB on the device | largest logit difference | greedy tokens equal | ms per decode step | min to max |
| --- | --- | --- | --- | --- | --- |
| whole model, ModelRunner | 2222.0 | reference | reference | 8.24 | 7.79 to 12.85 |
| 1 rank | 2222.0 | 0.0000 | 192 of 192 | 8.30 | 7.91 to 10.06 |
| 2 ranks | 2474.0 | 0.7500 | 192 of 192 | 11.27 | 10.35 to 12.24 |
| 4 ranks | 2978.0 | 0.6406 | 192 of 192 | 18.01 | 16.09 to 28.82 |

### Run 3

| layout | MiB on the device | largest logit difference | greedy tokens equal | ms per decode step | min to max |
| --- | --- | --- | --- | --- | --- |
| whole model, ModelRunner | 2222.0 | reference | reference | 8.27 | 7.83 to 10.03 |
| 1 rank | 2222.0 | 0.0000 | 192 of 192 | 8.34 | 7.85 to 11.22 |
| 2 ranks | 2474.0 | 0.7500 | 192 of 192 | 11.15 | 10.26 to 15.07 |
| 4 ranks | 2978.0 | 0.6406 | 192 of 192 | 17.49 | 15.11 to 26.80 |
