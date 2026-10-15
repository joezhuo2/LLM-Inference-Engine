import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import bench_csv as bc  # noqa: E402
import bench_workload as bw  # noqa: E402


class Workload(unittest.TestCase):
    def test_matches_the_pinned_tokens_of_the_cpp_workload(self):
        w = bw.make_workload(2, (4, 4), (1, 9))
        self.assertEqual(w[0]["prompt"], [1, 7689, 23406, 81])
        self.assertEqual(w[1]["prompt"], [1, 18454, 30427, 8986])
        self.assertEqual(w[0]["max_new_tokens"], 8)
        self.assertEqual(w[1]["max_new_tokens"], 4)

    def test_same_seed_same_workload_and_prompts_avoid_specials(self):
        a = bw.make_workload(20, (5, 30), (1, 20), request_rate=4.0, seed=11)
        b = bw.make_workload(20, (5, 30), (1, 20), request_rate=4.0, seed=11)
        self.assertEqual(a, b)
        for r in a:
            self.assertEqual(r["prompt"][0], 1)
            self.assertNotIn(1, r["prompt"][1:])
            self.assertNotIn(2, r["prompt"])
            self.assertTrue(all(0 <= t < 32000 for t in r["prompt"]))
        self.assertEqual(a[0]["arrival"], 0.0)
        self.assertTrue(all(x["arrival"] <= y["arrival"] for x, y in zip(a, a[1:])))

    def test_all_arrive_at_once_without_a_rate(self):
        self.assertEqual({r["arrival"] for r in bw.make_workload(10, (3, 3), (2, 2))}, {0.0})

    def test_length_ranges(self):
        self.assertEqual(bw.parse_length_range("128"), (128, 128))
        self.assertEqual(bw.parse_length_range("8:64"), (8, 64))
        for bad in ("0", "a", "9:3", "1:2:3", ""):
            with self.assertRaises(ValueError):
                bw.parse_length_range(bad)


class Metrics(unittest.TestCase):
    def test_percentile_interpolates_like_numpy(self):
        self.assertAlmostEqual(bc.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(bc.percentile([1, 2, 3, 4], 99), 3.97)
        self.assertTrue(math.isnan(bc.percentile([], 50)))

    def test_request_metrics_definitions(self):
        m = bc.request_metrics(3, 10, 5, arrival=1.0, first_token=1.5, finished=2.5)
        self.assertAlmostEqual(m["ttft_s"], 0.5)
        self.assertAlmostEqual(m["tpot_s"], 0.25)
        self.assertAlmostEqual(m["latency_s"], 1.5)
        one = bc.request_metrics(4, 10, 1, 0.0, 0.1, 0.1)
        self.assertTrue(math.isnan(one["tpot_s"]))

    def test_summary_and_csv_have_the_engine_column_names(self):
        ms = [bc.request_metrics(i, 8, 4, 0.0, 0.1, 0.4) for i in range(3)]
        s = bc.summarize(ms, wall_s=2.0)
        self.assertEqual(s["output_tokens"], 12)
        self.assertAlmostEqual(s["output_throughput"], 6.0)
        self.assertAlmostEqual(s["tpot_s_p50"], 0.1)
        b = bc.bandwidth_fields(s, weight_bytes=2e9)
        self.assertAlmostEqual(b["bandwidth_gb_s"], 20.0)
        with tempfile.TemporaryDirectory() as d:
            bc.write_request_csv(Path(d) / "r.csv", ms)
            bc.write_summary_csv(Path(d) / "s.csv", {"driver": "x", "max_batch": 1}, {**s, **b})
            header = (Path(d) / "r.csv").read_text().splitlines()[0]
            self.assertEqual(header, "id,prompt_tokens,output_tokens,preemptions,arrival_s,ttft_s,tpot_s,latency_s")
            first = (Path(d) / "s.csv").read_text().splitlines()[0].split(",")
            self.assertEqual(first[:2], ["driver", "max_batch"])
            self.assertIn("ttft_s_p99", first)

    def test_summarize_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bc.summarize([], 1.0)
        with self.assertRaises(ValueError):
            bc.summarize([bc.request_metrics(0, 1, 2, 0, 0.1, 0.2)], 0.0)


if __name__ == "__main__":
    unittest.main()
