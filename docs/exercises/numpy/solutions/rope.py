import numpy as np
from _load import load

exercise = load(__file__)


def rope(qkv, positions, heads, kv_heads, theta):
    head_dim = qkv.shape[1] // (heads + 2 * kv_heads)
    half = head_dim // 2
    inv_freq = theta ** (-2.0 * np.arange(half) / head_dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    out = qkv.copy()
    rotated = out[:, : (heads + kv_heads) * head_dim].reshape(len(qkv), heads + kv_heads, head_dim)
    lo = rotated[..., :half].copy()
    hi = rotated[..., half:].copy()
    rotated[..., :half] = lo * cos - hi * sin
    rotated[..., half:] = hi * cos + lo * sin
    return out


if __name__ == "__main__":
    exercise.check(rope)
