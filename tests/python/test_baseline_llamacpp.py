import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import baseline_llamacpp as bl  # noqa: E402

PP = {"build_commit": "abc1234", "build_number": 6000, "gpu_info": "NVIDIA GeForce RTX 5060 Laptop GPU",
      "backends": "CUDA", "model_filename": "/models/tinyllama-bf16.gguf", "n_prompt": 128, "n_gen": 0,
      "avg_ts": 8000.0, "samples_ts": [7900.0, 8000.0, 8100.0]}
TG = {"build_commit": "abc1234", "build_number": 6000, "gpu_info": "NVIDIA GeForce RTX 5060 Laptop GPU",
      "backends": "CUDA", "model_filename": "/models/tinyllama-bf16.gguf", "n_prompt": 0, "n_gen": 256,
      "avg_ts": 100.0, "samples_ts": [99.0, 100.0, 101.0]}


class LlamaBench(unittest.TestCase):
    def test_parse_requires_a_nonempty_list(self):
        with self.assertRaises(ValueError):
            bl.parse_llama_bench("[]")
        with self.assertRaises(ValueError):
            bl.parse_llama_bench("{}")
        self.assertEqual(len(bl.parse_llama_bench(json.dumps([PP, TG]))), 2)

    def test_summary_maps_throughput_to_ttft_and_tpot(self):
        s = bl.summary_from_rows([PP, TG], 128, 256)
        self.assertAlmostEqual(s["ttft_s_p50"], 128 / 8000.0)
        self.assertAlmostEqual(s["tpot_s_p50"], 1 / 100.0)
        self.assertEqual(s["requests"], 3)
        self.assertAlmostEqual(s["bandwidth_gb_s"], 2_069_024_768 / 0.01 / 1e9)
        self.assertGreater(s["latency_s_p50"], 255 * 0.01)

    def test_average_only_rows_are_accepted(self):
        pp = {k: v for k, v in PP.items() if k != "samples_ts"}
        tg = {k: v for k, v in TG.items() if k != "samples_ts"}
        s = bl.summary_from_rows([pp, tg], 128, 256)
        self.assertEqual(s["requests"], 1)
        self.assertAlmostEqual(s["tpot_s_p99"], 0.01)

    def test_both_tests_are_required(self):
        with self.assertRaises(ValueError):
            bl.summary_from_rows([PP], 128, 256)

    def test_throughput_comes_from_the_durations(self):
        tg = {**TG, "samples_ns": [2_560_000_000, 2_560_000_000, 2_600_000_000]}
        self.assertAlmostEqual(bl.samples(tg)[0], 100.0)

    def test_wrapped_negative_durations_are_dropped(self):
        tg = {**TG, "samples_ns": [18446744071733648849, 549156004, 543191289],
              "samples_ts": [3.46945e-09, 116.542, 117.822]}
        values = bl.samples(tg)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 256 * 1e9 / 549156004)
        with self.assertRaises(ValueError):
            bl.samples({**TG, "samples_ns": [18446744071733648849]})

    def test_main_writes_the_engine_summary_columns(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "lb.json"
            src.write_text(json.dumps([PP, TG]))
            out = Path(d) / "s.csv"
            self.assertEqual(bl.main(["--from-json", str(src), "--summary", str(out)]), 0)
            header, row = out.read_text().splitlines()
            cols = dict(zip(header.split(","), row.split(",")))
            self.assertEqual(cols["driver"], "llamacpp_bench")
            self.assertEqual(cols["max_batch"], "1")
            self.assertEqual(cols["model"], "tinyllama-bf16.gguf")
            self.assertFalse(math.isnan(float(cols["tpot_s_p50"])))


if __name__ == "__main__":
    unittest.main()
