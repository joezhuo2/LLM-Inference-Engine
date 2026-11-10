import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import batch_report as br  # noqa: E402


def write_summary(d, name, driver, max_batch, tpot, throughput, graph_steps="0"):
    row = {"date": "2026-11-09T10:00:00Z", "gpu": "GPU X", "driver": driver,
           "max_batch": max_batch, "num_requests": 2 * max_batch, "graph_steps": graph_steps,
           "tpot_s_p50": tpot, "output_throughput": throughput, "ttft_s_p50": 0.1, "steps": 70,
           "peak_kv_blocks": max_batch * 8}
    with open(d / f"{name}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


class BatchReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_takes_the_median_of_the_runs_per_batch_and_driver(self):
        for r, tpot in enumerate((0.010, 0.012, 0.030), 1):
            write_summary(self.dir, f"engine_b32_r{r}", "batched", 32, tpot, 1000 * r)
        write_summary(self.dir, "engine_graphs_b32_r1", "batched_graphs", 32, 0.009, 3000)
        rows = br.batch_rows(br.load(self.dir))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eager"]["runs"], 3)
        self.assertAlmostEqual(rows[0]["eager"]["tpot_s_p50"], 0.012)
        self.assertAlmostEqual(rows[0]["eager"]["output_throughput"], 2000)
        self.assertAlmostEqual(rows[0]["graphs"]["tpot_s_p50"], 0.009)

    def test_sorts_batches_numerically_and_leaves_missing_cells_empty(self):
        write_summary(self.dir, "engine_b128_r1", "batched", 128, 0.02, 6000)
        write_summary(self.dir, "engine_b32_r1", "batched", 32, 0.01, 3000)
        text = br.markdown(br.load(self.dir))
        lines = text.splitlines()
        self.assertTrue(lines[2].startswith("| 32 | 10.00 |  |"))
        self.assertTrue(lines[3].startswith("| 128 | 20.00 |  |"))

    def test_padding_runs_are_grouped_by_graph_steps_and_kept_out_of_the_sweep(self):
        write_summary(self.dir, "pad_steps1_r1", "batched_graphs", 32, 0.011, 1500, "1")
        write_summary(self.dir, "pad_steps8_r1", "batched_graphs", 32, 0.009, 1800, "8")
        runs = br.load(self.dir)
        self.assertEqual(br.batch_rows(runs), [])
        self.assertEqual([r["graph_steps"] for r in br.pad_rows(runs)], [1, 8])
        self.assertIn("| 8 | 9.00 | 1800 |", br.markdown(runs))


if __name__ == "__main__":
    unittest.main()
