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
            self.assertEqual(bl.main(["bench", "--from-json", str(src), "--summary", str(out)]), 0)
            header, row = out.read_text().splitlines()
            cols = dict(zip(header.split(","), row.split(",")))
            self.assertEqual(cols["driver"], "llamacpp_bench")
            self.assertEqual(cols["max_batch"], "1")
            self.assertEqual(cols["model"], "tinyllama-bf16.gguf")
            self.assertFalse(math.isnan(float(cols["tpot_s_p50"])))


def batched_row(pl, t_pp, t_tg, shared=0):
    return {"n_kv_max": 4096, "n_batch": 2048, "n_ubatch": 512, "flash_attn": -1, "is_pp_shared": shared,
            "n_gpu_layers": 99, "n_threads": 5, "n_threads_batch": 5, "pp": 64, "tg": 63, "pl": pl,
            "n_kv": pl * 127, "t_pp": t_pp, "speed_pp": pl * 64 / t_pp, "t_tg": t_tg,
            "speed_tg": pl * 63 / t_tg, "t": t_pp + t_tg, "speed": pl * 127 / (t_pp + t_tg)}


WARMUP = batched_row(1, 0.491913, 0.535171)
B1 = batched_row(1, 0.018006, 0.553058)
B16 = batched_row(16, 0.122363, 0.659913)


class LlamaBatchedBench(unittest.TestCase):
    def test_command_prepends_a_warmup_batch_and_sizes_the_context(self):
        cmd = bl.batched_command("bb", "m.gguf", 64, 64, [1, 4, 16], ["-ngl", "99"])
        self.assertEqual(cmd[cmd.index("-npl") + 1], "1,1,4,16")
        self.assertEqual(cmd[cmd.index("-ntg") + 1], "63")
        self.assertEqual(cmd[cmd.index("-c") + 1], str(16 * 128))
        self.assertEqual(cmd[-2:], ["-ngl", "99"])

    def test_parse_keeps_only_jsonl_rows(self):
        text = "noise\n" + "\n".join(json.dumps(r) for r in (WARMUP, B1, B16)) + "\n"
        self.assertEqual([r["pl"] for r in bl.parse_batched_bench(text)], [1, 1, 16])
        with self.assertRaises(ValueError):
            bl.parse_batched_bench("no results\n")
        with self.assertRaises(ValueError):
            bl.parse_batched_bench(json.dumps(batched_row(4, 0.03, 0.5, shared=1)))
        with self.assertRaises(ValueError):
            bl.parse_batched_bench(json.dumps({"pl": 1}))

    def test_a_batch_maps_to_requests_finishing_together(self):
        s = bl.summary_from_batched_row(B16, 16)
        self.assertEqual(s["requests"], 16)
        self.assertEqual(s["output_tokens"], 16 * 64)
        self.assertAlmostEqual(s["ttft_s_p50"], 0.122363)
        self.assertAlmostEqual(s["tpot_s_p50"], 0.659913 / 63)
        self.assertAlmostEqual(s["output_throughput"], 16 * 64 / B16["t"])
        self.assertNotIn("bandwidth_gb_s", s)
        self.assertAlmostEqual(bl.summary_from_batched_row(B1, 1)["bandwidth_gb_s"],
                               2_069_024_768 / (0.553058 / 63) / 1e9)

    def test_more_requests_queue_as_consecutive_batches(self):
        s = bl.summary_from_batched_row(B16, 32)
        one = bl.summary_from_batched_row(B16, 16)
        self.assertEqual(s["requests"], 32)
        self.assertAlmostEqual(s["wall_s"], 2 * B16["t"])
        self.assertAlmostEqual(s["output_throughput"], one["output_throughput"])
        self.assertAlmostEqual(s["tpot_s_p50"], one["tpot_s_p50"])
        self.assertAlmostEqual(s["ttft_s_mean"], B16["t_pp"] + B16["t"] / 2)
        self.assertAlmostEqual(s["latency_s_p99"], 2 * B16["t"])

    def test_main_drops_the_warmup_row_and_writes_one_summary_per_batch_and_run(self):
        with tempfile.TemporaryDirectory() as d:
            runs = []
            for i in (1, 2):
                src = Path(d) / f"bb{i}.jsonl"
                src.write_text("\n".join(json.dumps(r) for r in (WARMUP, B1, B16)) + "\n")
                runs.append(str(src))
            self.assertEqual(bl.main(["batched", "--from-jsonl", *runs, "--out-dir", d,
                                      "--num-requests", "32"]), 0)
            written = sorted(p.name for p in Path(d).glob("*_summary.csv"))
            self.assertEqual(written, ["llamacpp_b16_r1_summary.csv", "llamacpp_b16_r2_summary.csv",
                                       "llamacpp_b1_r1_summary.csv", "llamacpp_b1_r2_summary.csv"])
            header, row = (Path(d) / "llamacpp_b1_r1_summary.csv").read_text().splitlines()
            cols = dict(zip(header.split(","), row.split(",")))
            self.assertEqual(cols["driver"], "llamacpp_batched")
            self.assertEqual(cols["output_len"], "64")
            self.assertEqual(cols["num_requests"], "32")
            self.assertAlmostEqual(float(cols["ttft_s_p50"]), 0.018006 + 15.5 * B1["t"])


if __name__ == "__main__":
    unittest.main()
