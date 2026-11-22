import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prefix_cache_report as pr  # noqa: E402


def write_run(d, traces, name, ttfts, hit_rate, prefill_steps):
    summary = {"date": "2026-11-21T10:00:00Z", "gpu": "GPU X", "prompt_len": "64",
               "prefix_hit_rate": hit_rate, "output_throughput": 1000.0}
    with open(d / f"{name}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    with open(d / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ttft_s"])
        for i, ttft in enumerate(ttfts):
            w.writerow([i, ttft])
    steps = [{"kind": "prefill", "start_s": 0.0, "end_s": 0.5, "seqs": [0]}]
    steps += [{"kind": "prefill", "start_s": 1.0, "end_s": 1.0 + s, "seqs": [1]} for s in prefill_steps]
    steps.append({"kind": "decode", "start_s": 2.0, "end_s": 9.0, "seqs": [0, 1]})
    (traces / f"{name}.json").write_text(json.dumps({"steps": steps}))


class PrefixCacheReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.traces = self.dir / "traces"
        self.traces.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_splits_the_first_request_from_the_later_ones_and_takes_medians(self):
        write_run(self.dir, self.traces, "on_p512_q2_r1", [0.05, 0.01, 0.02, 0.03], 0.8, [0.004, 0.006])
        write_run(self.dir, self.traces, "on_p512_q2_r2", [0.07, 0.03, 0.04, 0.05], 0.8, [0.008])
        write_run(self.dir, self.traces, "off_p512_q2_r1", [0.05, 0.05, 0.06, 0.05], 0.0, [0.05])
        write_run(self.dir, self.traces, "on_p512_q0_r1", [0.05, 0.01, 0.02, 0.03], 0.5, [0.02])
        rows = pr.rows(pr.load(self.dir, self.traces))
        self.assertEqual([(r["rate"], r["cache"]) for r in rows], [(2, "on"), (2, "off"), (0, "on")])
        on = rows[0]
        self.assertEqual(on["runs"], 2)
        self.assertAlmostEqual(on["first_ttft"], 0.06)
        self.assertAlmostEqual(on["later_ttft"], 0.03)
        self.assertAlmostEqual(on["prefill"], 0.0065)
        text = pr.table(rows)
        self.assertIn("| 512 | 64 | 2/s | on | 60.0 | 30.0 | 6.5 | 80.0% | 1000.0 | 2 |", text)
        self.assertIn("| 512 | 64 | at once | on |", text)

    def test_runs_without_traces_leave_the_prefill_column_empty(self):
        write_run(self.dir, self.traces, "off_p256_q2_r1", [0.05, 0.05], 0.0, [0.05])
        (self.traces / "off_p256_q2_r1.json").unlink()
        text = pr.table(pr.rows(pr.load(self.dir, self.traces)))
        self.assertIn("| 256 | 64 | 2/s | off | 50.0 | 50.0 |  | 0.0% |", text)


if __name__ == "__main__":
    unittest.main()
