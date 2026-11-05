import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import onchip_report as orp  # noqa: E402

BENCH = """\
metric,unit,value,sm_mhz
register_file,KB/SM,256.00,
l2_latency,cycles,318.10,2812
l2_latency,ns,113.11,2812
"""


def rows(run, latency, mhz):
    stamp = {"run": run, "date": "2026-11-04T00:00:00Z", "commit": "abc1234", "gpu": "RTX"}
    return [{**r, **stamp} for r in orp.parse_bench(BENCH.replace("318.10", str(latency)).replace("2812", str(mhz)))]


class Parse(unittest.TestCase):
    def test_rows_without_a_clock_keep_none(self):
        parsed = orp.parse_bench(BENCH)
        self.assertEqual(parsed[0], {"metric": "register_file", "unit": "KB/SM", "value": 256.0, "sm_mhz": None})
        self.assertEqual(parsed[1]["sm_mhz"], 2812.0)
        self.assertAlmostEqual(parsed[2]["value"], 113.11)


class Table(unittest.TestCase):
    def setUp(self):
        self.rows = rows(1, 300, 2800) + rows(2, 330, 2700) + rows(3, 318, 2812)

    def test_median_and_spread_per_metric_and_unit(self):
        text = orp.table_markdown(self.rows)
        self.assertIn("3 runs", text)
        self.assertIn("commit abc1234 on the RTX", text)
        self.assertIn("| L2 | hit latency, pointer chase in 4 MB | cycles | 318 | 300 to 330 | 2,800 |", text)
        self.assertIn("| registers | register file | KB/SM | 256 | 256 to 256 |  |", text)

    def test_unknown_metrics_keep_their_name(self):
        extra = [{**self.rows[0], "metric": "new_thing", "unit": "x", "value": 1.5, "sm_mhz": None}]
        self.assertIn("|  | new_thing | x | 1.50 | 1.50 to 1.50 |  |", orp.table_markdown(self.rows + extra))

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "onchip.csv"
            orp.write_rows(path, self.rows)
            self.assertEqual(orp.read_rows(path), self.rows)

    def test_from_csv_writes_the_table(self):
        with tempfile.TemporaryDirectory() as d:
            saved = Path(d) / "saved.csv"
            orp.write_rows(saved, self.rows)
            out = Path(d) / "out"
            self.assertEqual(orp.main(["--from-csv", str(saved), "--out-dir", str(out)]), 0)
            self.assertEqual((out / "onchip.md").read_text(), orp.table_markdown(self.rows))


if __name__ == "__main__":
    unittest.main()
