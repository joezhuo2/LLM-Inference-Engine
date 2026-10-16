import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import make_results as mr  # noqa: E402

LOG = """\
compare          identical (64 tokens)
definition       differs at step 28: ours 304, HF 363, 1 BF16 ulps apart in HF's logits
sql              differs at step 12: ours 5, HF 6, 2 BF16 ulps apart in HF's logits
free-running greedy identical to HF on 15 of 20 prompts
teacher-forced argmax 4651/4700 (98.96%), mean |dlogit| 0.0312
"""


def row(system, batch, throughput=0.0, bandwidth=float("nan")):
    return {"system": system, "max_batch": float(batch), "output_throughput": throughput,
            "bandwidth_gb_s": bandwidth}


class Correctness(unittest.TestCase):
    def test_parse(self):
        c = mr.parse_correctness(LOG)
        self.assertEqual(c["argmax_matches"], 4651)
        self.assertEqual(c["argmax_total"], 4700)
        self.assertAlmostEqual(c["mean_abs_logit"], 0.0312)
        self.assertEqual((c["exact_prompts"], c["prompts"]), (15, 20))
        self.assertEqual((c["near_ties"], c["max_tie_ulps"]), (2, 2))

    def test_empty_log(self):
        self.assertEqual(mr.parse_correctness("nothing here"), {"near_ties": 0, "max_tie_ulps": 0})

    def test_environment(self):
        env = mr.parse_environment("gpu = RTX\ncommit=abc\nnoise\n")
        self.assertEqual(env, {"gpu": "RTX", "commit": "abc"})


class Scoreboard(unittest.TestCase):
    def test_verdicts_follow_the_targets(self):
        rows = [row("engine", 1, 100.0, 227.0), row("engine", 16, 1000.0)]
        text = mr.scoreboard(rows, mr.parse_correctness(LOG))
        lines = {l.split("|")[1].strip(): l for l in text.splitlines()[2:]}
        self.assertTrue(lines["1a"].rstrip().endswith("| not met |"))
        self.assertIn("98.96%", lines["1a"])
        self.assertTrue(lines["1b"].rstrip().endswith("| not met |"))
        self.assertIn("at most 2 BF16 ulps", lines["1b"])
        self.assertTrue(lines["2"].rstrip().endswith("| met |"))
        self.assertIn("70%", lines["2"])
        self.assertTrue(lines["3"].rstrip().endswith("| met |"))
        self.assertIn("10.0x", lines["3"])

    def test_missing_data_is_reported_not_guessed(self):
        text = mr.scoreboard([], {})
        self.assertEqual(text.count("not recorded"), 4)


class Blocks(unittest.TestCase):
    def test_replace_block_keeps_the_rest(self):
        doc = "a\n<!-- auto:x -->\nold\n<!-- /auto:x -->\nb\n"
        self.assertEqual(mr.replace_block(doc, "x", "new\n"),
                         "a\n<!-- auto:x -->\nnew\n<!-- /auto:x -->\nb\n")

    def test_missing_markers_raise(self):
        with self.assertRaises(ValueError):
            mr.replace_block("nothing", "x", "y")

    def test_main_updates_a_document(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "results").mkdir()
            (d / "results" / "environment.txt").write_text("gpu=GPU X\ncommit=abc123\n")
            (d / "results" / "correctness.txt").write_text(LOG)
            (d / "results" / "engine_b1_summary.csv").write_text(
                "driver,max_batch,output_throughput,tpot_s_p50,ttft_s_p50,gpu\nbatched,1,100,0.01,0.1,GPU X\n")
            (d / "results" / "decode_step.md").write_text("## Launch gaps\n\n- decode: 1 steps\n")
            doc = d / "results.md"
            doc.write_text("".join(f"<!-- auto:{n} -->\n<!-- /auto:{n} -->\n"
                                   for n in ("environment", "profile", "scoreboard", "table")))
            self.assertEqual(mr.main(["--results-dir", str(d / "results"), "--doc", str(doc)]), 0)
            out = doc.read_text()
            self.assertIn("- GPU: GPU X", out)
            self.assertIn("| engine | 1 | 100.0 |", out)
            self.assertIn("| 2 | Batch-1 decode bandwidth", out)
            self.assertIn("### Launch gaps", out)


if __name__ == "__main__":
    unittest.main()
