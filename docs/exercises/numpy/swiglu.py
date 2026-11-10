import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 1e-4


def swiglu(gate_up):
    raise NotImplementedError


def reference(gate_up):
    ff = gate_up.shape[1] // 2
    out = np.zeros((gate_up.shape[0], ff))
    for t in range(gate_up.shape[0]):
        for j in range(ff):
            gate = float(gate_up[t, j])
            up = float(gate_up[t, ff + j])
            out[t, j] = gate / (1.0 + math.exp(-gate)) * up
    return out


def check(fn=None):
    fn = fn or swiglu
    g = c.rng(7)
    errors = []
    for ff in (c.INTERMEDIATE, 7):
        gate_up = (3.0 * g.standard_normal((3, 2 * ff))).astype(np.float32)
        errors.append(c.max_error(fn(gate_up.copy()), reference(gate_up)))
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
