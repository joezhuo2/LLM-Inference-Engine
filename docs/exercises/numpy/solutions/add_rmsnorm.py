import numpy as np
from _load import load

exercise = load(__file__)


def add_rmsnorm(x, residual, weight, eps):
    total = x + residual
    rstd = 1.0 / np.sqrt(np.mean(total * total, axis=1, keepdims=True) + eps)
    return total, weight * (total * rstd)


if __name__ == "__main__":
    exercise.check(add_rmsnorm)
