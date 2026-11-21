import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

TOLERANCE = 0.015
DRAWS = 20000


def top_k_threshold(scaled, k):
    raise NotImplementedError(
        "Write top_k_threshold with NumPy and no Python loop: return the k-th largest of the scaled logits, so that keeping every token at or above it keeps the top k, ties at the cut included. reference_threshold() below is the slow version your result is checked against; the spec is on this exercise's card on the exercises page."
    )


def gumbel_max(logits, temperature, u, top_k=0):
    raise NotImplementedError(
        "Write gumbel_max with NumPy and no Python loop: at temperature 0 return the argmax of the logits (greedy); otherwise divide the logits by the temperature, drop every token below top_k_threshold() when 0 < top_k < vocab, add the Gumbel noise -log(-log(u)) and return the argmax. reference() below is the slow version your result is checked against; the spec is on this exercise's card on the exercises page."
    )


def reference_threshold(scaled, k):
    return sorted((float(v) for v in scaled), reverse=True)[k - 1]


def reference(logits, temperature, u, top_k=0):
    if temperature == 0:
        best = 0
        for i in range(len(logits)):
            if logits[i] > logits[best]:
                best = i
        return best
    scaled = [float(v) / temperature for v in logits]
    cut = reference_threshold(scaled, top_k) if 0 < top_k < len(logits) else -math.inf
    best, best_score = 0, -math.inf
    for i, s in enumerate(scaled):
        score = s - math.log(-math.log(float(u[i]))) if s >= cut else -math.inf
        if score > best_score:
            best, best_score = i, score
    return best


def uniform(g, shape):
    return (g.integers(0, 2**32, size=shape) + 0.5) / 2**32


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def check(fn=None, threshold=None):
    fn = fn or gumbel_max
    threshold = threshold or top_k_threshold
    g = c.rng(12)
    for _ in range(300):
        logits = np.round(3.0 * g.standard_normal(50), 1)
        u = uniform(g, 50)
        temperature = float(g.choice([0.0, 0.5, 1.0, 1.7]))
        top_k = int(g.choice([0, 1, 5, 50]))
        if temperature > 0 and 0 < top_k < len(logits):
            got, want = threshold(logits / temperature, top_k), reference_threshold(logits / temperature, top_k)
            if got != want:
                c.fail(f"top_k_threshold gave {got} for k = {top_k}, the reference gave {want}")
        got, want = int(fn(logits.copy(), temperature, u.copy(), top_k)), reference(logits, temperature, u, top_k)
        if got != want:
            c.fail(f"with the same u you chose token {got} at temperature {temperature} and top-k {top_k}, the reference chose {want}")
    logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0, -3.0])
    errors = []
    for temperature, top_k in ((1.0, 0), (0.6, 0), (1.3, 3)):
        counts = np.zeros(len(logits))
        for u in uniform(g, (DRAWS, len(logits))):
            counts[int(fn(logits.copy(), temperature, u, top_k))] += 1
        kept = logits / temperature if top_k == 0 else np.where(logits >= np.sort(logits)[-top_k], logits / temperature, -np.inf)
        errors.append(c.max_error(counts / DRAWS, softmax(kept)))
    c.report(c.worst(*errors), TOLERANCE)


if __name__ == "__main__":
    check()
