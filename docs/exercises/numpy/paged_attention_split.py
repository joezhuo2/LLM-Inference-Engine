import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 1e-3


def split_ranges(context_len, num_splits, block_size):
    total_blocks = -(-context_len // block_size)
    per_split = -(-total_blocks // num_splits)
    return [(min(context_len, s * per_split * block_size), min(context_len, (s + 1) * per_split * block_size)) for s in range(num_splits)]


def paged_attention_split(qkv, k_cache, v_cache, block_tables, context_lens, heads, num_splits):
    raise NotImplementedError


def check(fn=None):
    fn = fn or paged_attention_split
    g = c.rng(5)
    qkv, k_cache, v_cache, tables, lens = c.paged_case(g, [1, 17, 100, 300, 0, 100, 100, 40])
    c.ramp(qkv, k_cache, tables, lens, 5, c.HEADS, True)
    c.ramp(qkv, k_cache, tables, lens, 6, c.HEADS, False)
    qkv[7, : c.HEADS * c.HEAD_DIM] *= 300
    want = c.reference_attention(qkv, k_cache, v_cache, tables, lens, c.HEADS)
    errors = []
    for num_splits in (1, 3, 16, 64):
        got = fn(qkv.copy(), k_cache.copy(), v_cache.copy(), tables.copy(), lens.copy(), c.HEADS, num_splits)
        errors.append(c.max_error(got, want))
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
