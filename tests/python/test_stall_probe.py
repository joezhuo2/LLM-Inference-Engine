import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import stall_probe as sp  # noqa: E402


def request(prompt, arrival, ttft, latency):
    return {"prompt_tokens": str(prompt), "arrival_s": str(arrival), "ttft_s": str(ttft), "latency_s": str(latency)}


class Prompts(unittest.TestCase):
    def test_the_long_prompt_comes_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = Path(tmp) / "p.json"
            short.write_text(json.dumps({"prompts": [{"name": str(i), "messages": []} for i in range(5)]}))
            prompts = sp.prompts_with_a_long_one(short, 4, 3)["prompts"]
        self.assertEqual([p["name"] for p in prompts], ["0", "1", "2", "long"])
        self.assertEqual(prompts[-1]["messages"][0]["content"], sp.SENTENCE * 3)

    def test_budget_zero_keeps_whole_prompt_prefill(self):
        cmd = sp.engine_command("e", "m", "p.json", 16, 16.0, 128, 0, "o.csv", "t.json")
        self.assertNotIn("--token-budget", cmd)
        self.assertEqual(cmd[cmd.index("--trace") + 1], "t.json")
        cmd = sp.engine_command("e", "m", "p.json", 16, 16.0, 128, 512, "o.csv", "t.json")
        self.assertEqual(cmd[cmd.index("--token-budget") + 1], "512")


class StallRun(unittest.TestCase):
    def test_counts_decoders_and_reads_the_long_prompt(self):
        requests = [request(30, 0.0, 0.1, 2.0), request(30, 0.9, 0.05, 2.0), request(30, 0.0, 0.1, 0.5),
                    request(1900, 1.0, 1.5, 3.0)]
        steps = [{"seqs": [1], "end_s": 1.0}, {"seqs": [1], "end_s": 1.02}, {"seqs": [1, 4], "end_s": 2.5}]
        result = sp.stall_run(requests, steps)
        self.assertEqual(result["decoding"], 2)
        self.assertEqual(result["long_prompt_tokens"], 1900)
        self.assertAlmostEqual(result["long_ttft_ms"], 1500.0)
        self.assertAlmostEqual(result["gap_ms_max"], 1480.0)

    def test_table_has_one_row_per_budget(self):
        rows = [{"token_budget": b, "run": r, "decoding": 15, "long_prompt_tokens": 1911, "long_ttft_ms": t,
                 "gap_ms_p99": t, "gap_ms_max": t, "date": "d", "commit": "c"}
                for b, r, t in ((0, 1, 1000.0), (0, 2, 1200.0), (0, 3, 1100.0), (512, 1, 300.0))]
        text = sp.table_markdown(rows)
        self.assertIn("a 1911-token prompt", text)
        self.assertIn("| none | 1911 | 15 | 1100 | 1100 | 1100 | 3 |", text)
        self.assertIn("| 512 | 1911 | 15 | 300 | 300 | 300 | 1 |", text)


if __name__ == "__main__":
    unittest.main()
