from _load import load

exercise = load(__file__)


def write_kv(k_cache, v_cache, qkv, slot_mapping):
    kv_heads, block_size, head_dim = k_cache.shape[1:]
    kv_width = kv_heads * head_dim
    width = qkv.shape[1]
    keys = qkv[:, width - 2 * kv_width : width - kv_width].reshape(-1, kv_heads, head_dim)
    values = qkv[:, width - kv_width :].reshape(-1, kv_heads, head_dim)
    live = slot_mapping >= 0
    blocks = slot_mapping[live] // block_size
    offsets = slot_mapping[live] % block_size
    k_cache[blocks, :, offsets, :] = keys[live]
    v_cache[blocks, :, offsets, :] = values[live]


if __name__ == "__main__":
    exercise.check(write_kv)
