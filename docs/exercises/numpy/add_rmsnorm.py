import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 1e-4


def add_rmsnorm(x, residual, weight, eps):
    raise NotImplementedError


def reference(x, residual, weight, eps):
    total = x.astype(np.float64) + residual.astype(np.float64)
    normed = np.zeros_like(total)
    for t in range(total.shape[0]):
        mean_square = sum(float(v) * float(v) for v in total[t]) / total.shape[1]
        normed[t] = weight.astype(np.float64) * total[t] / np.sqrt(mean_square + eps)
    return total, normed


def check(fn=None):
    fn = fn or add_rmsnorm
    g = c.rng(6)
    errors = []
    for hidden in (c.HIDDEN, 100):
        x = g.standard_normal((5, hidden)).astype(np.float32)
        residual = g.standard_normal((5, hidden)).astype(np.float32)
        weight = g.standard_normal(hidden).astype(np.float32)
        x[2], residual[2] = 0.0, 0.0
        x[3], residual[3] = 1e-3 * x[3], 1e-3 * residual[3]
        want_sum, want_norm = reference(x, residual, weight, c.RMS_EPS)
        got_sum, got_norm = fn(x.copy(), residual.copy(), weight.copy(), c.RMS_EPS)
        errors += [c.max_error(got_sum, want_sum), c.max_error(got_norm, want_norm)]
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
