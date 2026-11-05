import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import load_report as lr  # noqa: E402


def write_run(d, name, driver, rate, wall, requests, preemptions="0", budget="0"):
    summary = {"version": "0.1.0" if driver.startswith("batched") else "vllm 0.32.0",
               "date": f"2026-11-0{rate}T10:00:00Z", "gpu": "GPU X",
               "driver": driver, "request_rate": rate, "wall_s": wall, "preemptions": preemptions,
               "token_budget": budget,
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

    def test_a_token_budget_is_its_own_system(self):
        self.assertEqual(lr.system_of({"driver": "batched_graphs", "token_budget": "512"}),
                         "engine + graphs, budget 512")
        self.assertEqual(lr.system_of({"driver": "batched_graphs", "token_budget": "0"}), "engine + graphs")
        self.assertEqual(lr.system_of({"driver": "vllm_online"}), "vLLM")

    def test_budgets_sort_after_their_engine_and_before_vllm(self):
        systems = ["vLLM", "engine + graphs, budget 2048", "engine + graphs", "engine + graphs, budget 512"]
        self.assertEqual(sorted(systems, key=lr.series_order),
                         ["engine + graphs", "engine + graphs, budget 512", "engine + graphs, budget 2048",
                          "vLLM"])

    def test_token_gaps_follow_each_sequence_through_the_steps(self):
        steps = [{"seqs": [1], "end_s": 0.1}, {"seqs": [], "end_s": 0.2}, {"seqs": [2], "end_s": 0.3},
                 {"seqs": [1, 2], "end_s": 0.5}, {"seqs": [1], "end_s": 0.6}]
        self.assertEqual([round(g, 6) for g in lr.token_gaps(steps)], [0.4, 0.2, 0.1])

    def test_nearest_rank(self):
        self.assertEqual(lr.nearest_rank(list(range(1, 101)), 0.99), 99)
        self.assertEqual(lr.nearest_rank([5.0], 0.99), 5.0)

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

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_main_writes_the_table_and_plots(self):
        for rate in (1, 2):
            write_run(self.dir, f"engine_q{rate}_r1", "batched_graphs", rate, 10.0, [(0.1, 0.02)])
        write_run(self.dir, "vllm_q1_r1", "vllm_online", 1, 10.0, [(0.9, 0.02)], preemptions="")
        self.assertEqual(lr.main(["--dir", str(self.dir), "--commit", "abc1234"]), 0)
        table = (self.dir / "load_table.md").read_text()
        self.assertTrue(table.startswith("Engine runs measured 2026-11-01 to 2026-11-02 at commit abc1234 on GPU X; "
                                         "vLLM runs measured 2026-11-01 (vllm 0.32.0).\n\n"))
        self.assertIn("| engine + graphs | 2 | 500 | 100 | 400 | 20.0 | 30.0 | 100% | 0.10 | 0 | 1 |",
                      table)
        self.assertIn("| vLLM | 1 | 500 | 100 | 400 | 20.0 | 30.0 | 0% | 0.00 |  | 1 |", table)
        for name in ("ttft_p99_vs_qps", "tpot_p99_vs_qps", "goodput_vs_qps"):
            self.assertTrue((self.dir / f"{name}.png").stat().st_size > 0)
        self.assertFalse((self.dir / "token_gaps.csv").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_traces_add_the_gap_table(self):
        traces = self.dir / "traces"
        traces.mkdir()
        write_run(self.dir, "engine_budget512_q2_r3", "batched_graphs", 2, 10.0, [(0.1, 0.02)], budget="512")
        write_run(self.dir, "vllm_q2_r1", "vllm_online", 2, 10.0, [(0.1, 0.02)], preemptions="")
        steps = [{"seqs": [1], "end_s": 0.0}, {"seqs": [1], "end_s": 0.01}, {"seqs": [1], "end_s": 0.25}]
        (traces / "engine_budget512_q2_r3.json").write_text(json.dumps({"steps": steps}))
        self.assertEqual(lr.main(["--dir", str(self.dir), "--traces", str(traces)]), 0)
        with open(self.dir / "token_gaps.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["system"], rows[0]["rate"], rows[0]["run"], rows[0]["gaps"]),
                         ("engine + graphs, budget 512", "2", "3", "2"))
        self.assertEqual(rows[0]["gap_ms_max"], "240.000")
        table = (self.dir / "load_table.md").read_text()
        self.assertIn("| engine + graphs, budget 512 | 2 | 125.0 | 240.0 | 240 | 1 |", table)
        self.assertTrue((self.dir / "gap_max_vs_qps.png").stat().st_size > 0)
        self.assertEqual(lr.main(["--dir", str(self.dir)]), 0)
        self.assertIn("largest gap ms", (self.dir / "load_table.md").read_text())


if __name__ == "__main__":
    unittest.main()
