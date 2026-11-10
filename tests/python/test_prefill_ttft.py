import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prefill_ttft as pt  # noqa: E402


def row(system, length, run, ttft):
    return {"system": system, "prompt_tokens": length, "run": run, "ttft_s": ttft,
            "date": "2026-11-03T00:00:00Z", "commit": "abc1234"}


class Commands(unittest.TestCase):
    def test_engine_serves_one_request_of_one_token(self):
        cmd = pt.engine_command("engine", "m", 512, 3, 2, "o.csv", False)
        self.assertEqual(cmd[cmd.index("--num-requests") + 1], "1")
        self.assertEqual(cmd[cmd.index("--output-len") + 1], "1")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "3")
        self.assertNotIn("--graphs", cmd)
        self.assertIn("--graphs", pt.engine_command("engine", "m", 512, 3, 2, "o.csv", True))
        self.assertNotIn("--token-budget", cmd)

    def test_a_budget_adds_the_flag(self):
        cmd = pt.engine_command("engine", "m", 2048, 1, 2, "o.csv", True, 512)
        self.assertIn("--graphs", cmd)
        self.assertEqual(cmd[cmd.index("--token-budget") + 1], "512")

    def test_the_paged_series_asks_for_the_old_prefill_attention(self):
        cmd = pt.engine_command("engine", "m", 2048, 1, 2, "o.csv", False, 0, True)
        self.assertEqual(cmd[cmd.index("--prefill-attention") + 1], "paged")
        self.assertNotIn("--prefill-attention", pt.engine_command("engine", "m", 2048, 1, 2, "o.csv", False))

    def test_the_paged_series_comes_first(self):
        rows = [row(pt.ENGINE, 100, 1, 0.1), row(pt.PAGED, 100, 1, 0.2)]
        text = pt.table_markdown(rows)
        self.assertLess(text.index(f"| {pt.PAGED} |"), text.index("| engine |"))

    def test_llama_context_fits_the_longest_prompt_and_its_token(self):
        cmd = pt.llama_command("bin", "m", [128, 2048], 512)
        self.assertEqual(cmd[cmd.index("-c") + 1], "2049")
        self.assertEqual(cmd[cmd.index("-b") + 1], "2048")
        self.assertEqual(cmd[cmd.index("-ub") + 1], "512")
        self.assertEqual(cmd[cmd.index("-npp") + 1], "128,2048")


class LlamaRows(unittest.TestCase):
    def test_the_first_row_of_each_length_is_warmup(self):
        rows = [{"pp": 128, "t_pp": 9.0}, {"pp": 128, "t_pp": 0.1}, {"pp": 512, "t_pp": 9.0}, {"pp": 512, "t_pp": 0.4}]
        self.assertEqual(pt.llama_ttfts(rows), {128: 0.1, 512: 0.4})


class Table(unittest.TestCase):
    def test_median_spread_and_per_token(self):
        rows = [row(pt.ENGINE, 100, r, t) for r, t in ((1, 0.1), (2, 0.2), (3, 0.6))]
        text = pt.table_markdown(rows)
        self.assertIn("at commit abc1234", text)
        self.assertIn("median of 3 runs", text)
        self.assertIn("| engine | 200.0 (100.0 to 600.0) |", text)
        self.assertIn("| engine | 2.000 |", text)

    def test_systems_keep_their_order(self):
        rows = [row(pt.LLAMA, 100, 1, 0.1), row(pt.ENGINE, 100, 1, 0.2)]
        text = pt.table_markdown(rows)
        self.assertLess(text.index("| engine |"), text.index("| llama.cpp -ub 512 |"))

    def test_a_budget_row_follows_its_engine(self):
        budget = f"{pt.GRAPHS}, budget 512"
        rows = [row(pt.LLAMA, 100, 1, 0.1), row(budget, 100, 1, 0.3), row(pt.GRAPHS, 100, 1, 0.2)]
        text = pt.table_markdown(rows)
        self.assertLess(text.index("| engine + graphs |"), text.index(f"| {budget} |"))
        self.assertLess(text.index(f"| {budget} |"), text.index("| llama.cpp -ub 512 |"))


class RoundTrip(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_from_csv_redraws_every_output(self):
        rows = [row(pt.ENGINE, n, 1, n * 1e-3) for n in (128, 512)] + [row(pt.LLAMA_FULL, n, 1, 0.1) for n in (128, 512)]
        rows += [row(f"{pt.GRAPHS}, budget 512", n, 1, 0.2) for n in (128, 512)]
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "saved.csv"
            pt.write_rows(saved, rows)
            self.assertEqual(pt.read_rows(saved), rows)
            out = Path(tmp) / "out"
            self.assertEqual(pt.main(["--from-csv", str(saved), "--out-dir", str(out)]), 0)
            for name in ("prefill_ttft.csv", "prefill.md", "prefill_ttft.png"):
                self.assertTrue((out / name).exists())


if __name__ == "__main__":
    unittest.main()
