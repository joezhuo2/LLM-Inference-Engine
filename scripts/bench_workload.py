import math

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        limit = MASK - MASK % n
        x = self.next()
        while x >= limit:
            x = self.next()
        return x % n

    def between(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)

    def unit(self) -> float:
        return (self.next() >> 11) * 2.0**-53


def parse_length_range(text: str):
    parts = text.split(":")
    if len(parts) > 2 or not all(p.isdigit() and int(p) >= 1 for p in parts):
        raise ValueError(f"length range '{text}': expected N or MIN:MAX with positive integers")
    lo = int(parts[0])
    hi = int(parts[-1])
    if hi < lo:
        raise ValueError(f"length range '{text}': MIN is larger than MAX")
    return lo, hi


def make_workload(num_requests, prompt_len, output_len, vocab=32000, bos_id=1, eos_id=2,
                  request_rate=0.0, seed=0):
    if num_requests < 1:
        raise ValueError("workload: num_requests must be positive")
    if vocab < 3 or bos_id == eos_id:
        raise ValueError("workload: bos_id and eos_id must be distinct ids in the vocab")
    rng = SplitMix64(seed)
    requests = []
    t = 0.0
    specials = sorted((bos_id, eos_id))
    for i in range(num_requests):
        max_new_tokens = rng.between(*output_len)
        prompt = [0] * rng.between(*prompt_len)
        prompt[0] = bos_id
        for j in range(1, len(prompt)):
            token = rng.below(vocab - 2)
            for special in specials:
                if token >= special:
                    token += 1
            prompt[j] = token
        if i > 0 and request_rate > 0.0:
            t -= math.log1p(-rng.unit()) / request_rate
        requests.append({"id": i, "prompt": prompt, "max_new_tokens": max_new_tokens, "arrival": t})
    return requests
