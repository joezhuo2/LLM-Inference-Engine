import numpy as np
from _load import load

exercise = load(__file__)


def top_k_threshold(scaled, k):
    return np.partition(scaled, -k)[-k]


def gumbel_max(logits, temperature, u, top_k=0):
    if temperature == 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    if 0 < top_k < len(logits):
        scaled = np.where(scaled >= top_k_threshold(scaled, top_k), scaled, -np.inf)
    return int(np.argmax(scaled - np.log(-np.log(u))))


if __name__ == "__main__":
    exercise.check(gumbel_max, top_k_threshold)
