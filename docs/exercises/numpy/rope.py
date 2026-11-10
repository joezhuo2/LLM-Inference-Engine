import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 2e-3


def rope(qkv, positions, heads, kv_heads, theta):
    raise NotImplementedError


def reference(qkv, positions, heads, kv_heads, theta):
    head_dim = qkv.shape[1] // (heads + 2 * kv_heads)
    half = head_dim // 2
    out = qkv.astype(np.float64)
    for t, position in enumerate(positions):
        for h in range(heads + kv_heads):
            for i in range(half):
                angle = float(position) * theta ** (-2.0 * i / head_dim)
                lo = float(qkv[t, h * head_dim + i])
                hi = float(qkv[t, h * head_dim + i + half])
                out[t, h * head_dim + i] = lo * math.cos(angle) - hi * math.sin(angle)
                out[t, h * head_dim + i + half] = hi * math.cos(angle) + lo * math.sin(angle)
    return out


def check(fn=None):
    fn = fn or rope
    g = c.rng(0)
    qkv = g.standard_normal((16, (c.HEADS + 2 * c.KV_HEADS) * c.HEAD_DIM)).astype(np.float32)
    positions = np.concatenate([[0, 1, 2047], g.integers(0, 2048, size=13)]).astype(np.int32)
    want = reference(qkv, positions, c.HEADS, c.KV_HEADS, c.ROPE_THETA)
    got = fn(qkv.copy(), positions.copy(), c.HEADS, c.KV_HEADS, c.ROPE_THETA)
    c.report(c.max_error(got, want), TOLERANCE)


if __name__ == "__main__":
    check()
