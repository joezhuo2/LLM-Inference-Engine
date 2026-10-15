import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import baseline_hf as bh  # noqa: E402


def request(i, prompt_len, new_tokens, arrival=0.0):
    return {"id": i, "prompt": list(range(prompt_len)), "max_new_tokens": new_tokens, "arrival": arrival}


class RunGrouping(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.original = bh.greedy_batch

        def fake(model, torch, prompts, steps, device):
            self.calls.append((len(prompts), steps))
            base = time.perf_counter()
            times = [[base + 0.001 * (k + 1) for k in range(steps)] for _ in prompts]
            return [[0] * steps for _ in prompts], times

        bh.greedy_batch = fake

    def tearDown(self):
        bh.greedy_batch = self.original

    def test_sequential_runs_one_request_per_call(self):
        reqs = [request(i, 4, 3) for i in range(3)]
        m = bh.run(None, None, reqs, 1, "cpu", time.perf_counter())
        self.assertEqual(self.calls, [(1, 3)] * 3)
        self.assertEqual([x["output_tokens"] for x in m], [3, 3, 3])

    def test_static_batches_run_to_the_longest_and_report_each_requests_own_length(self):
        reqs = [request(0, 4, 2), request(1, 4, 5), request(2, 4, 3)]
        m = bh.run(None, None, reqs, 2, "cpu", time.perf_counter())
        self.assertEqual(self.calls, [(2, 5), (1, 3)])
        self.assertEqual([x["output_tokens"] for x in m], [2, 5, 3])
        self.assertLess(m[0]["latency_s"], m[1]["latency_s"])

    def test_a_group_waits_for_its_last_arrival(self):
        start = time.perf_counter()
        reqs = [request(0, 4, 2, 0.0), request(1, 4, 2, 0.05)]
        bh.run(None, None, reqs, 2, "cpu", start)
        self.assertGreaterEqual(time.perf_counter() - start, 0.05)

    def test_times_are_relative_to_the_start(self):
        m = bh.run(None, None, [request(0, 4, 3)], 1, "cpu", time.perf_counter())
        self.assertAlmostEqual(m[0]["arrival_s"], 0.0)
        self.assertGreater(m[0]["ttft_s"], 0.0)
        self.assertLess(m[0]["latency_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
