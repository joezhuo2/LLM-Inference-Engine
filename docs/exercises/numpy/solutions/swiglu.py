import numpy as np
from _load import load

exercise = load(__file__)


def swiglu(gate_up):
    gate, up = np.split(gate_up, 2, axis=1)
    return gate / (1.0 + np.exp(-gate)) * up


if __name__ == "__main__":
    exercise.check(swiglu)
