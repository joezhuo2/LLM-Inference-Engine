import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_rmsnorm
import common as c
import rope
import swiglu

TOLERANCE = 1e-3
KV_HEADS = 2
HEADS = KV_HEADS * (c.HEADS // c.KV_HEADS)
HEAD_DIM = 8
HIDDEN = HEADS * HEAD_DIM
INTERMEDIATE = HIDDEN * c.INTERMEDIATE // c.HIDDEN
VOCAB = 500


def decode_step(token, position, k_cache, v_cache, w):
    raise NotImplementedError(
        "Write one decode step for one token with NumPy operations on whole arrays: look up its embedding row, run the one layer in the chapter's order (RMSNorm, QKV projection, RoPE on Q and K, put the new key and value after the cached ones, attention, output projection, add, RMSNorm, gate/up, SwiGLU, down, add), then the final norm, the LM head and the argmax, and return the logits and the chosen id. reference() below is the slow version your result is checked against; the spec is on this exercise's card on the exercises page."
    )


def matvec(x, weight):
    return np.array([sum(float(a) * float(b) for a, b in zip(x, weight[:, j])) for j in range(weight.shape[1])])


def norm(x, weight):
    return add_rmsnorm.reference(x[None], np.zeros((1, len(x))), weight, c.RMS_EPS)[1][0]


def reference(token, position, k_cache, v_cache, w):
    q_end, k_end = HEADS * HEAD_DIM, (HEADS + KV_HEADS) * HEAD_DIM
    x = w["embed"][token].astype(np.float64)
    qkv = rope.reference(matvec(norm(x, w["ln1"]), w["wqkv"])[None], [position], HEADS, KV_HEADS, c.ROPE_THETA)[0]
    keys = np.concatenate([k_cache, qkv[q_end:k_end].reshape(KV_HEADS, 1, HEAD_DIM)], axis=1)
    values = np.concatenate([v_cache, qkv[k_end:].reshape(KV_HEADS, 1, HEAD_DIM)], axis=1)
    attn = c.reference_attention(qkv[None], keys[None], values[None], np.zeros((1, 1), dtype=np.int32), [position + 1], HEADS)[0]
    x, h = add_rmsnorm.reference(x[None], matvec(attn, w["wo"])[None], w["ln2"], c.RMS_EPS)
    x = x[0] + matvec(swiglu.reference(matvec(h[0], w["wgu"])[None])[0], w["wdown"])
    logits = matvec(norm(x, w["final_norm"]), w["lm_head"])
    return logits, int(np.argmax(logits))


def model(g):
    def matrix(rows, cols):
        return (g.standard_normal((rows, cols)) / np.sqrt(rows)).astype(np.float32)

    def scale():
        return (1.0 + 0.1 * g.standard_normal(HIDDEN)).astype(np.float32)

    return {"embed": g.standard_normal((VOCAB, HIDDEN)).astype(np.float32), "ln1": scale(), "wqkv": matrix(HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM), "wo": matrix(HEADS * HEAD_DIM, HIDDEN), "ln2": scale(), "wgu": matrix(HIDDEN, 2 * INTERMEDIATE), "wdown": matrix(INTERMEDIATE, HIDDEN), "final_norm": scale(), "lm_head": matrix(HIDDEN, VOCAB)}


def check(fn=None):
    fn = fn or decode_step
    g = c.rng(8)
    w = model(g)
    errors = []
    for position in (0, 9):
        token = int(g.integers(VOCAB))
        k_cache = g.standard_normal((KV_HEADS, position, HEAD_DIM)).astype(np.float32)
        v_cache = g.standard_normal((KV_HEADS, position, HEAD_DIM)).astype(np.float32)
        want_logits, want_id = reference(token, position, k_cache, v_cache, w)
        got_logits, got_id = fn(token, position, k_cache.copy(), v_cache.copy(), {k: v.copy() for k, v in w.items()})
        errors.append(c.max_error(got_logits, want_logits))
        if int(got_id) != want_id:
            c.fail(f"at position {position} you chose token {int(got_id)}, the reference chose {want_id}")
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
