import numpy as np
from _load import load

exercise = load(__file__)


def paged_attention_v2(qkv, k_cache, v_cache, block_tables, context_lens, heads, chunk):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    group = heads // kv_heads
    out = np.zeros((qkv.shape[0], heads * head_dim), dtype=np.float32)
    for b, length in enumerate(context_lens):
        if length == 0:
            continue
        for h in range(heads):
            q = qkv[b, h * head_dim : (h + 1) * head_dim]
            running_max = -np.inf
            running_sum = 0.0
            acc = np.zeros(head_dim)
            for start in range(0, length, chunk):
                positions = np.arange(start, min(start + chunk, length))
                blocks = block_tables[b, positions // block_size]
                offsets = positions % block_size
                scores = k_cache[blocks, h // group, offsets] @ q / np.sqrt(head_dim)
                new_max = max(running_max, scores.max())
                rescale = np.exp(running_max - new_max)
                weights = np.exp(scores - new_max)
                running_sum = running_sum * rescale + weights.sum()
                acc = acc * rescale + weights @ v_cache[blocks, h // group, offsets]
                running_max = new_max
            out[b, h * head_dim : (h + 1) * head_dim] = acc / running_sum
    return out


if __name__ == "__main__":
    exercise.check(paged_attention_v2)
