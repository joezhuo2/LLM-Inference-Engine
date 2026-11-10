import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c


def write_kv(k_cache, v_cache, qkv, slot_mapping):
    raise NotImplementedError


def reference(k_cache, v_cache, qkv, slot_mapping):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    width = qkv.shape[1]
    kv_width = kv_heads * head_dim
    k_start = width - 2 * kv_width
    v_start = width - kv_width
    for t, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block, offset = divmod(int(slot), block_size)
        for h in range(kv_heads):
            k_cache[block, h, offset, :] = qkv[t, k_start + h * head_dim : k_start + (h + 1) * head_dim]
            v_cache[block, h, offset, :] = qkv[t, v_start + h * head_dim : v_start + (h + 1) * head_dim]


def check(fn=None):
    fn = fn or write_kv
    g = c.rng(1)
    num_blocks, tokens = 8, 40
    num_slots = num_blocks * c.BLOCK_SIZE
    pool = g.permutation(num_slots)
    pool = np.concatenate([[0, num_slots - 1], pool[(pool != 0) & (pool != num_slots - 1)]])
    slots = pool[:tokens].astype(np.int32)
    slots[[5, 20]] = -1
    qkv = g.standard_normal((tokens, (c.HEADS + 2 * c.KV_HEADS) * c.HEAD_DIM)).astype(np.float32)
    shape = (num_blocks, c.KV_HEADS, c.BLOCK_SIZE, c.HEAD_DIM)
    k_cache = g.standard_normal(shape).astype(np.float32)
    v_cache = g.standard_normal(shape).astype(np.float32)
    want_k, want_v = k_cache.copy(), v_cache.copy()
    reference(want_k, want_v, qkv, slots)
    got_k, got_v = k_cache.copy(), v_cache.copy()
    fn(got_k, got_v, qkv.copy(), slots.copy())
    c.report(c.worst(c.max_error(got_k, want_k), c.max_error(got_v, want_v)), 0.0)


if __name__ == "__main__":
    check()
