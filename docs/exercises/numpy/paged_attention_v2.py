import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 1e-3


def paged_attention_v2(qkv, k_cache, v_cache, block_tables, context_lens, heads, chunk):
    raise NotImplementedError


def check(fn=None):
    fn = fn or paged_attention_v2
    g = c.rng(4)
    qkv, k_cache, v_cache, tables, lens = c.paged_case(g, [1, 17, 100, 300, 0, 100, 100, 40])
    c.ramp(qkv, k_cache, tables, lens, 5, c.HEADS, True)
    c.ramp(qkv, k_cache, tables, lens, 6, c.HEADS, False)
    qkv[7, : c.HEADS * c.HEAD_DIM] *= 300
    want = c.reference_attention(qkv, k_cache, v_cache, tables, lens, c.HEADS)
    errors = []
    for chunk in (3, 16, 256):
        got = fn(qkv.copy(), k_cache.copy(), v_cache.copy(), tables.copy(), lens.copy(), c.HEADS, chunk)
        errors.append(c.max_error(got, want))
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
