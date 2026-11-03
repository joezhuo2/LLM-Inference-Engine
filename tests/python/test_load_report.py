import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import load_report as lr  # noqa: E402


def write_run(d, name, driver, rate, wall, requests, preemptions="0"):
    summary = {"driver": driver, "request_rate": rate, "wall_s": wall, "preemptions": preemptions,
               "ttft_s_p50": 0.1, "ttft_s_p99": 0.4, "tpot_s_p50": 0.02, "tpot_s_p99": 0.03,
               "output_throughput": 500}
    with open(d / f"{name}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    with open(d / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ttft_s", "tpot_s"])
        w.writeheader()
        for ttft, tpot in requests:
            w.writerow({"ttft_s": ttft, "tpot_s": tpot})


class LoadReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_slo_needs_both_limits(self):
        self.assertTrue(lr.meets_slo({"ttft_s": "0.5", "tpot_s": "0.05"}, 500, 50))
        self.assertFalse(lr.meets_slo({"ttft_s": "0.6", "tpot_s": "0.01"}, 500, 50))
        self.assertFalse(lr.meets_slo({"ttft_s": "0.1", "tpot_s": "0.06"}, 500, 50))

    def test_one_token_request_has_no_tpot_to_miss(self):
        self.assertTrue(lr.meets_slo({"ttft_s": "0.1", "tpot_s": "nan"}, 500, 50))

    def test_goodput_counts_only_requests_meeting_the_slo(self):
        write_run(self.dir, "engine_q2_r1", "batched_graphs", 2, 10.0,
                  [(0.1, 0.02), (0.1, 0.02), (0.9, 0.02), (0.1, 0.09)])
        run = lr.load_run(self.dir / "engine_q2_r1_summary.csv", 500, 50)
        self.assertEqual(run["system"], "engine + graphs")
        self.assertEqual(run["rate"], 2)
        self.assertAlmostEqual(run["attainment"], 0.5)
        self.assertAlmostEqual(run["goodput"], 0.2)

    def test_aggregate_takes_the_median_per_system_and_rate(self):
        base = {"system": "vLLM", "rate": 1.0, "attainment": 1.0, "preemptions": math.nan,
                "ttft_s_p50": 0.1, "ttft_s_p99": 0.2, "tpot_s_p50": 0.1, "tpot_s_p99": 0.2,
                "output_throughput": 100.0}
        runs = [{**base, "goodput": g} for g in (1.0, 2.0, 9.0)]
        runs.append({**base, "rate": 2.0, "goodput": 5.0})
        points = lr.aggregate(runs)
        self.assertEqual([(p["rate"], p["goodput"], p["runs"]) for p in points],
                         [(1.0, 2.0, 3), (2.0, 5.0, 1)])
        self.assertTrue(math.isnan(points[0]["preemptions"]))

    def test_main_writes_the_table_and_plots(self):
        for rate in (1, 2):
            write_run(self.dir, f"engine_q{rate}_r1", "batched_graphs", rate, 10.0, [(0.1, 0.02)])
        write_run(self.dir, "vllm_q1_r1", "vllm_online", 1, 10.0, [(0.9, 0.02)], preemptions="")
        self.assertEqual(lr.main(["--dir", str(self.dir)]), 0)
        table = (self.dir / "load_table.md").read_text()
        self.assertIn("| engine + graphs | 2 | 500 | 100 | 400 | 20.0 | 30.0 | 100% | 0.10 | 0 | 1 |",
                      table)
        self.assertIn("| vLLM | 1 | 500 | 100 | 400 | 20.0 | 30.0 | 0% | 0.00 |  | 1 |", table)
        for name in ("ttft_p99_vs_qps", "tpot_p99_vs_qps", "goodput_vs_qps"):
            self.assertTrue((self.dir / f"{name}.png").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
