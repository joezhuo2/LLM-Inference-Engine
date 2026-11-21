import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c


def weight_bytes():
    layer = c.HIDDEN * (c.HEADS + 2 * c.KV_HEADS) * c.HEAD_DIM + c.HIDDEN * c.HIDDEN + c.HIDDEN * 2 * c.INTERMEDIATE + c.INTERMEDIATE * c.HIDDEN
    return 2 * (c.LAYERS * layer + c.VOCAB * c.HIDDEN)


def best_copy(megabytes, repeats):
    src = np.ones(megabytes * 2**20 // 4, dtype=np.float32)
    dst = np.zeros_like(src)
    np.copyto(dst, src)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - start)
    if not np.array_equal(dst, src):
        c.fail("the copy is wrong")
    return 2 * src.nbytes / best / 1e9, src.nbytes


def main():
    megabytes = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    repeats = max(5, min(200, 2**31 // (megabytes * 2**20)))
    gbs, size = best_copy(megabytes, repeats)
    weights = weight_bytes()
    print(f"copied {size / 1e6:,.0f} MB into another array {repeats} times, after one warm-up copy")
    print(f"best copy: {gbs:.1f} GB/s, counting the bytes read and the bytes written")
    print("This is a floor for your machine's memory bandwidth, not its peak: NumPy copies on one thread, and memory usually needs several cores at once to reach its peak.")
    print(f"TinyLlama reads every weight once per decoded token, all {c.LAYERS} layers and then the LM head: {weights / 1e9:.2f} GB at 2 bytes per weight.")
    floor_ms = weights / (gbs * 1e9) * 1e3
    print(f"At {gbs:.1f} GB/s reading them takes {floor_ms:,.0f} ms. A decoded token cannot take less than its bytes divided by the bandwidth, so at the rate NumPy reached, one token on this CPU would take at least {floor_ms:,.0f} ms; your memory's real peak would lower that bound.")


if __name__ == "__main__":
    main()
