import numpy as np
from _load import load

exercise = load(__file__)


def paged_attention(qkv, k_cache, v_cache, block_tables, context_lens, heads):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    group = heads // kv_heads
    out = np.zeros((qkv.shape[0], heads * head_dim), dtype=np.float32)
    for b, length in enumerate(context_lens):
        if length == 0:
            continue
        positions = np.arange(length)
        blocks = block_tables[b, positions // block_size]
        offsets = positions % block_size
        for h in range(heads):
            q = qkv[b, h * head_dim : (h + 1) * head_dim]
            keys = k_cache[blocks, h // group, offsets]
            values = v_cache[blocks, h // group, offsets]
            scores = keys @ q / np.sqrt(head_dim)
            weights = np.exp(scores - scores.max())
            out[b, h * head_dim : (h + 1) * head_dim] = (weights / weights.sum()) @ values
    return out


if __name__ == "__main__":
    exercise.check(paged_attention)
