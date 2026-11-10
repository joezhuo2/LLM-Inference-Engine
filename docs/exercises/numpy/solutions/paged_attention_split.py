import numpy as np
from _load import load

exercise = load(__file__)


def paged_attention_split(qkv, k_cache, v_cache, block_tables, context_lens, heads, num_splits):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    group = heads // kv_heads
    out = np.zeros((qkv.shape[0], heads * head_dim), dtype=np.float32)
    for b, length in enumerate(context_lens):
        if length == 0:
            continue
        ranges = [(start, end) for start, end in exercise.split_ranges(length, num_splits, block_size) if start < end]
        for h in range(heads):
            q = qkv[b, h * head_dim : (h + 1) * head_dim]
            partials = []
            for start, end in ranges:
                positions = np.arange(start, end)
                blocks = block_tables[b, positions // block_size]
                offsets = positions % block_size
                scores = k_cache[blocks, h // group, offsets] @ q / np.sqrt(head_dim)
                split_max = scores.max()
                weights = np.exp(scores - split_max)
                partials.append((split_max, weights.sum(), weights @ v_cache[blocks, h // group, offsets]))
            global_max = max(m for m, _, _ in partials)
            scales = [np.exp(m - global_max) for m, _, _ in partials]
            total = sum(s * l for s, (_, l, _) in zip(scales, partials))
            acc = sum(s * a for s, (_, _, a) in zip(scales, partials))
            out[b, h * head_dim : (h + 1) * head_dim] = acc / total
    return out


if __name__ == "__main__":
    exercise.check(paged_attention_split)
