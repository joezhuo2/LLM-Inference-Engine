import numpy as np
from _load import load

exercise = load(__file__)
c = exercise.c
HEADS, KV_HEADS, HEAD_DIM = exercise.HEADS, exercise.KV_HEADS, exercise.HEAD_DIM


def rmsnorm(x, weight):
    return weight * x / np.sqrt(np.mean(x * x) + c.RMS_EPS)


def rotate(heads, position):
    half = HEAD_DIM // 2
    angle = position * c.ROPE_THETA ** (-2.0 * np.arange(half) / HEAD_DIM)
    cos, sin = np.cos(angle), np.sin(angle)
    lo, hi = heads[:, :half], heads[:, half:]
    return np.concatenate([lo * cos - hi * sin, hi * cos + lo * sin], axis=1)


def attention(q, keys, values):
    group = HEADS // KV_HEADS
    keys, values = np.repeat(keys, group, axis=0), np.repeat(values, group, axis=0)
    scores = np.einsum("hd,htd->ht", q, keys) / np.sqrt(HEAD_DIM)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ht,htd->hd", weights, values).reshape(-1)


def decode_step(token, position, k_cache, v_cache, w):
    x = w["embed"][token]
    q, k, v = np.split(rmsnorm(x, w["ln1"]) @ w["wqkv"], [HEADS * HEAD_DIM, (HEADS + KV_HEADS) * HEAD_DIM])
    q = rotate(q.reshape(HEADS, HEAD_DIM), position)
    k = rotate(k.reshape(KV_HEADS, HEAD_DIM), position)
    keys = np.concatenate([k_cache, k[:, None]], axis=1)
    values = np.concatenate([v_cache, v.reshape(KV_HEADS, 1, HEAD_DIM)], axis=1)
    x = x + attention(q, keys, values) @ w["wo"]
    gate, up = np.split(rmsnorm(x, w["ln2"]) @ w["wgu"], 2)
    x = x + (gate / (1.0 + np.exp(-gate)) * up) @ w["wdown"]
    logits = rmsnorm(x, w["final_norm"]) @ w["lm_head"]
    return logits, int(np.argmax(logits))


if __name__ == "__main__":
    exercise.check(decode_step)
