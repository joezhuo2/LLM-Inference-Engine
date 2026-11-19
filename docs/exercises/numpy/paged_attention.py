import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 1e-3


def paged_attention(qkv, k_cache, v_cache, block_tables, context_lens, heads):
    raise NotImplementedError(
        "Write paged attention in NumPy: for each row and head, gather its keys and values through the block table and score them as whole arrays, with no Python loop over positions. reference_attention() in common.py is the slow version your result is checked against; the spec is on this exercise's card on the exercises page."
    )


def check(fn=None):
    fn = fn or paged_attention
    g = c.rng(2)
    qkv, k_cache, v_cache, tables, lens = c.paged_case(g, [1, 15, 16, 17, 37, 0, 40])
    qkv[6, : c.HEADS * c.HEAD_DIM] *= 300
    want = c.reference_attention(qkv, k_cache, v_cache, tables, lens, c.HEADS)
    got = fn(qkv.copy(), k_cache.copy(), v_cache.copy(), tables.copy(), lens.copy(), c.HEADS)
    c.report(c.max_error(got, want), TOLERANCE)


if __name__ == "__main__":
    check()
