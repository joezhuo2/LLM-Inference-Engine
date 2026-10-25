import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import roofline as rl  # noqa: E402

TINYLLAMA = {
    "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22, "num_attention_heads": 32,
    "num_key_value_heads": 4, "vocab_size": 32000,
}

PROFILE = """## Launch gaps per forward step

- decode: 252 steps

## GPU time per op in a decode step

| op | kernels/step | GPU us/step | % of kernel time |
| --- | --- | --- | --- |
| gate_up | 22.0 | 3161.5 | 43.9 |
| attention | 22.0 | 317.1 | 4.4 |
| rope | 22.0 | 59.5 | 0.8 |

## Top 12 kernels

| kernel | count | total ms | avg us | % |
| --- | --- | --- | --- | --- |
| gemv | 22432 | 1688.4 | 75.3 | 90.8 |
"""


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.shape = rl.model_shape(TINYLLAMA)

    def test_shape_derives_head_dim(self):
        self.assertEqual(self.shape["head_dim"], 64)
        self.assertEqual(self.shape["kv_heads"], 4)

    def test_gemm_weights_match_the_decode_weight_bytes(self):
        costs = rl.op_costs(self.shape, 1, 0)
        params = sum(c["flops"] for name, c in costs.items() if name != "attention") / 2
        norms = 2048 * (2 * 22 + 1)
        self.assertEqual(int(params + norms) * 2, 2_069_024_768)

    def test_batch_one_gemm_is_about_one_flop_per_byte(self):
        c = rl.op_costs(self.shape, 1, 96)["gate_up"]
        self.assertAlmostEqual(c["flops"] / c["bytes"], 1.0, places=2)

    def test_batching_raises_gemm_intensity_almost_linearly(self):
        c = rl.op_costs(self.shape, 16, 96)["down"]
        self.assertGreater(c["flops"] / c["bytes"], 15.0)
        self.assertLess(c["flops"] / c["bytes"], 16.0)

    def test_attention_reads_the_kv_cache_once_per_row(self):
        c = rl.op_costs(self.shape, 2, 100)["attention"]
        self.assertEqual(c["flops"], 4.0 * 2 * 2048 * 100 * 22)
        self.assertEqual(c["bytes"], 2.0 * 2 * (2 * 256 * 100 + 2 * 2048) * 22)

    def test_attention_intensity_does_not_grow_with_rows(self):
        one = rl.op_costs(self.shape, 1, 500)["attention"]
        many = rl.op_costs(self.shape, 16, 500)["attention"]
        self.assertAlmostEqual(one["flops"] / one["bytes"], many["flops"] / many["bytes"])

    def test_roof_is_bandwidth_then_compute(self):
        self.assertAlmostEqual(rl.roof(1.0, 300.0, 30.0), 300e9)
        self.assertAlmostEqual(rl.roof(1000.0, 300.0, 30.0), 30e12)


class ProfileTest(unittest.TestCase):
    def test_parse_op_table_reads_only_the_op_section(self):
        times = rl.parse_op_table(PROFILE)
        self.assertEqual(times, {"gate_up": 3161.5, "attention": 317.1, "rope": 59.5})

    def test_parse_op_table_accepts_demoted_headings(self):
        self.assertIn("gate_up", rl.parse_op_table(PROFILE.replace("## ", "### ")))

    def test_op_points_skip_ops_without_a_cost_model(self):
        shape = rl.model_shape(TINYLLAMA)
        points = rl.op_points(shape, [(1, PROFILE)], 96, 323.6, 34.5)
        self.assertEqual([p["name"] for p in points], ["gate_up", "attention"])
        gate_up = points[0]
        self.assertAlmostEqual(gate_up["gb_s"], gate_up["bytes"] / 3161.5e-6 / 1e9)
        self.assertAlmostEqual(gate_up["pct_of_roof"], 100.0 * gate_up["gb_s"] / 323.6, places=0)

    def test_profile_argument(self):
        self.assertEqual(rl.parse_profile_arg("a/b:c.md:16"), (Path("a/b:c.md"), 16))
        with self.assertRaises(Exception):
            rl.parse_profile_arg("results/decode_step.md")


class StepTest(unittest.TestCase):
    def test_step_points_take_the_median_tpot_per_system_and_batch(self):
        shape = rl.model_shape(TINYLLAMA)
        rows = [{"system": "engine", "max_batch": 1.0, "tpot_s_p50": t} for t in (0.009, 0.008, 0.010)]
        rows.append({"system": "engine + graphs", "max_batch": 4.0, "tpot_s_p50": 0.008})
        rows.append({"system": "engine", "max_batch": 2.0, "tpot_s_p50": math.nan})
        points = rl.step_points(shape, rows, 96, 323.6, 34.5)
        self.assertEqual([(p["name"], p["rows"]) for p in points], [("engine + graphs", 4), ("engine", 1)])
        self.assertAlmostEqual(points[1]["us"], 9000.0)
        self.assertAlmostEqual(points[0]["intensity"], 4.0, delta=0.05)


class MainTest(unittest.TestCase):
    def test_writes_plot_and_table(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "config.json").write_text(json.dumps(TINYLLAMA))
            (d / "step.md").write_text(PROFILE)
            summary = d / "engine_graphs_b1_r1_summary.csv"
            with open(summary, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["driver", "max_batch", "tpot_s_p50", "gpu"])
                w.writerow(["batched_graphs", 1, 0.0078, "Test GPU"])
            rl.main([str(summary), "--profile", f"{d / 'step.md'}:1", "--profile", f"{d / 'none.md'}:16",
                     "--config", str(d / "config.json"), "--out-dir", str(d / "plots")])
            self.assertGreater((d / "plots" / "roofline.png").stat().st_size, 0)
            table = (d / "plots" / "roofline.md").read_text()
            self.assertIn("ridge point 107 FLOP/byte", table)
            self.assertIn("| gate_up | 1 | 1.0 | 3161.5 |", table)
            self.assertIn("| engine + graphs | 1 | 1.0 | 7800 |", table)


if __name__ == "__main__":
    unittest.main()
