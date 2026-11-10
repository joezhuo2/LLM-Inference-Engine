import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

_config = json.loads(re.search(r'kTinyLlama = R"\((.*?)\)";', (ROOT / "tests" / "unit" / "test_config.cpp").read_text(), re.S).group(1))

HIDDEN = _config["hidden_size"]
INTERMEDIATE = _config["intermediate_size"]
HEADS = _config["num_attention_heads"]
KV_HEADS = _config["num_key_value_heads"]
HEAD_DIM = HIDDEN // HEADS
RMS_EPS = _config["rms_norm_eps"]
ROPE_THETA = _config["rope_theta"]
BLOCK_SIZE = 16


def rng(seed):
    return np.random.default_rng(seed)


def fail(message):
    print(message)
    raise SystemExit(1)


def max_error(got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        fail(f"wrong shape {got.shape}, expected {want.shape}")
    return float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))


def worst(*errors):
    return float(np.max(errors))


def report(error, tolerance):
    if error <= tolerance:
        print("ok")
        return
    fail(f"max error {error:.3g}, must be at most {tolerance:g}")


def paged_case(generator, lengths, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, block_size=BLOCK_SIZE, spare=3):
    needed = [-(-n // block_size) for n in lengths]
    num_blocks = sum(needed) + spare
    width = max(needed + [1])
    physical = generator.permutation(num_blocks)
    k_cache = np.full((num_blocks, kv_heads, block_size, head_dim), np.nan, dtype=np.float32)
    v_cache = np.full((num_blocks, kv_heads, block_size, head_dim), np.nan, dtype=np.float32)
    tables = np.full((len(lengths), width), -1, dtype=np.int32)
    used = 0
    for row, (n, count) in enumerate(zip(lengths, needed)):
        ids = physical[used : used + count]
        used += count
        tables[row, :count] = ids
        for j in range(n):
            k_cache[ids[j // block_size], :, j % block_size] = generator.standard_normal((kv_heads, head_dim))
            v_cache[ids[j // block_size], :, j % block_size] = generator.standard_normal((kv_heads, head_dim))
    qkv = generator.standard_normal((len(lengths), (heads + 2 * kv_heads) * head_dim)).astype(np.float32)
    return qkv, k_cache, v_cache, tables, np.array(lengths, dtype=np.int32)


def ramp(qkv, k_cache, block_tables, context_lens, row, heads, rising):
    block_size, head_dim = k_cache.shape[2], k_cache.shape[3]
    n = int(context_lens[row])
    qkv[row, : heads * head_dim] = 1.0
    for j in range(n):
        k_cache[block_tables[row, j // block_size], :, j % block_size] = 0.02 * (j if rising else n - 1 - j)


def reference_attention(qkv, k_cache, v_cache, block_tables, context_lens, heads):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    group = heads // kv_heads
    out = np.zeros((qkv.shape[0], heads * head_dim))
    for b, n in enumerate(context_lens):
        if n == 0:
            continue
        blocks = block_tables[b, : -(-n // block_size)]
        keys = np.concatenate([k_cache[i] for i in blocks], axis=1)[:, :n].astype(np.float64)
        values = np.concatenate([v_cache[i] for i in blocks], axis=1)[:, :n].astype(np.float64)
        for h in range(heads):
            q = qkv[b, h * head_dim : (h + 1) * head_dim].astype(np.float64)
            scores = keys[h // group] @ q / np.sqrt(head_dim)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[b, h * head_dim : (h + 1) * head_dim] = weights @ values[h // group]
    return out
