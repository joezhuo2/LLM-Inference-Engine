import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import profile_report as pr  # noqa: E402


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
        """
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)",
                     [(1, "gemv"), (2, "attn"), (3, "qkv"), (4, "forward_decode"), (5, "attention")])
    conn.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
        [
            (0, 1000, 7, None, 4),
            (10, 100, 7, "qkv", None),
            (100, 200, 7, None, 5),
            (2000, 3000, 7, "update", None),
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
        [(20, 25, 7, 1), (120, 125, 7, 2), (2100, 2105, 7, 3)],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
        [(50, 150, 1, 1, 1), (300, 400, 2, 2, 2), (5000, 5100, 1, 1, 3)],
    )
    return conn


class BusyAndSpan(unittest.TestCase):
    def test_overlap_counts_once_and_gap_is_reported(self):
        busy, span, gaps = pr.busy_and_span([(0, 10), (5, 20), (30, 40)])
        self.assertEqual(busy, 30)
        self.assertEqual(span, 40)
        self.assertEqual(gaps, [10])

    def test_single_kernel_has_no_gap(self):
        self.assertEqual(pr.busy_and_span([(5, 9)]), (4, 4, []))


class Attribution(unittest.TestCase):
    def setUp(self):
        conn = make_db()
        strings = pr.string_table(conn)
        self.kernels = pr.attribute(pr.load_kernels(conn, strings), pr.load_ranges(conn, strings))

    def test_kernels_get_the_innermost_op_and_their_step(self):
        ops = sorted((k["launch"], k["op"], k["kind"]) for k in self.kernels)
        self.assertEqual(ops[0], (20, "qkv", "forward_decode"))
        self.assertEqual(ops[1], (120, "attention", "forward_decode"))

    def test_a_kernel_outside_every_step_is_not_in_a_step(self):
        outside = [k for k in self.kernels if k["launch"] == 2100][0]
        self.assertIsNone(outside["step"])

    def test_step_table_spans_first_start_to_last_end(self):
        rows = pr.step_table(self.kernels)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["kernels"], 2)
        self.assertAlmostEqual(row["span_us"], 0.35)
        self.assertAlmostEqual(row["busy_us"], 0.2)
        self.assertAlmostEqual(row["idle_pct"], 100 * 150 / 350)

    def test_op_table_sums_decode_kernel_time(self):
        rows = {r["op"]: r for r in pr.op_table(self.kernels)}
        self.assertAlmostEqual(rows["qkv"]["us_per_step"], 0.1)
        self.assertAlmostEqual(rows["attention"]["us_per_step"], 0.1)
        self.assertAlmostEqual(rows["qkv"]["pct"], 50.0)

    def test_kernel_table_sorts_by_total_time(self):
        rows = pr.kernel_table(self.kernels, 5)
        self.assertEqual([r["kernel"] for r in rows], ["gemv", "attn"])
        self.assertEqual(rows[0]["count"], 2)


class Bandwidth(unittest.TestCase):
    def test_achieved_bandwidth_uses_the_median_tpot(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["max_batch", "tpot_s_p50"])
                w.writerow(["1", "0.01"])
            b = pr.bandwidth_section(path, 200.0, 2e9, 8000.0)
        self.assertAlmostEqual(b["achieved_gb_s"], 200.0)
        self.assertAlmostEqual(b["pct_of_peak"], 100.0)
        self.assertAlmostEqual(b["busy_pct_of_tpot"], 80.0)
        self.assertAlmostEqual(b["kernel_gb_s"], 250.0)

    def test_report_runs_on_a_capture_without_steps(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            """
        )
        self.assertIn("no forward_decode", pr.report(conn))


if __name__ == "__main__":
    unittest.main()
