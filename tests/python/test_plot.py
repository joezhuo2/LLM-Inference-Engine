import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import plot  # noqa: E402


def write_summary(path, driver, max_batch, throughput, tpot, bandwidth=None, gpu="GPU X"):
    row = {"driver": driver, "max_batch": max_batch, "output_throughput": throughput,
           "tpot_s_p50": tpot, "ttft_s_p50": 0.1, "gpu": gpu}
    if bandwidth is not None:
        row["bandwidth_gb_s"] = bandwidth
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


class Plot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        write_summary(d / "a.csv", "batched", 1, 100.0, 0.0100)
        write_summary(d / "b.csv", "batched", 16, 900.0, 0.0150)
        write_summary(d / "c.csv", "batched", 16, 1100.0, 0.0160)
        write_summary(d / "d.csv", "hf_static_batch", 16, 500.0, 0.0300)
        write_summary(d / "e.csv", "llamacpp_bench", 1, 120.0, 0.0080, bandwidth=258.0)
        self.paths = sorted(d.glob("*.csv"))
        self.rows = plot.load_summaries(self.paths)

    def tearDown(self):
        self.tmp.cleanup()

    def test_system_names(self):
        self.assertEqual(plot.system_of("one_at_a_time"), "engine")
        self.assertEqual(plot.system_of("batched"), "engine")
        self.assertEqual(plot.system_of("batched_graphs"), "engine + graphs")
        self.assertEqual(plot.system_of("hf_sequential"), "HF transformers")
        self.assertEqual(plot.system_of("llamacpp_bench"), "llama.cpp")

    def test_batch_one_bandwidth_is_derived_from_tpot_when_missing(self):
        engine1 = [r for r in self.rows if r["system"] == "engine" and r["max_batch"] == 1][0]
        self.assertAlmostEqual(engine1["bandwidth_gb_s"], plot.DECODE_WEIGHT_BYTES / 0.01 / 1e9)
        engine16 = [r for r in self.rows if r["system"] == "engine" and r["max_batch"] == 16][0]
        self.assertTrue(math.isnan(engine16["bandwidth_gb_s"]))
        llama = [r for r in self.rows if r["system"] == "llama.cpp"][0]
        self.assertEqual(llama["bandwidth_gb_s"], 258.0)

    def test_repeated_runs_aggregate_to_the_median_in_system_order(self):
        s = plot.series(self.rows, "output_throughput")
        self.assertEqual(list(s), ["engine", "HF transformers", "llama.cpp"])
        self.assertEqual(s["engine"], [(1, 100.0), (16, 1000.0)])

    def test_table_lists_every_system_and_batch_once(self):
        lines = plot.table_markdown(self.rows).strip().splitlines()
        self.assertEqual(len(lines), 2 + 4)
        self.assertTrue(lines[2].startswith("| engine | 1 |"))
        self.assertIn("| engine | 16 | 1000.0 | 15.50 |", lines[3])
        self.assertTrue(lines[-1].endswith("| 1 |"))

    def test_close_end_labels_are_spread_around_their_values(self):
        out = plot.spread_labels([1264.0, 1284.0, 1402.0, 397.0], 60.0)
        self.assertAlmostEqual(out[0], 1244.0)
        self.assertAlmostEqual(out[1], 1304.0)
        self.assertEqual(out[2:], [1402.0, 397.0])
        chain = plot.spread_labels([100.0, 110.0, 150.0], 60.0)
        self.assertAlmostEqual(min(b - a for a, b in zip(chain, chain[1:])), 60.0)
        self.assertAlmostEqual(sum(chain) / 3, 120.0)

    def test_main_writes_the_images_and_the_table(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        out = Path(self.tmp.name) / "plots"
        self.assertEqual(plot.main([*map(str, self.paths), "--out-dir", str(out), "--theme", "dark"]), 0)
        for name in ("throughput_vs_batch.png", "tpot_vs_batch.png", "batch1_bandwidth.png",
                     "summary_table.md"):
            self.assertTrue((out / name).stat().st_size > 0, name)


if __name__ == "__main__":
    unittest.main()
