#!/usr/bin/env python3
import argparse
import csv
import io
import statistics
import subprocess
import sys
import time
from pathlib import Path

COLUMNS = ["metric", "unit", "run", "value", "sm_mhz", "date", "commit", "gpu"]
LABELS = {
    "register_file": ("registers", "register file"),
    "fma_latency": ("registers", "dependent FMA latency"),
    "smem_size": ("shared memory", "size"),
    "smem_latency": ("shared memory", "load latency, pointer chase"),
    "smem_bandwidth_32bit": ("shared memory", "read bandwidth, 32-bit loads"),
    "smem_bandwidth_128bit": ("shared memory", "read bandwidth, 128-bit loads"),
    "l2_size": ("L2", "size"),
    "l2_latency": ("L2", "hit latency, pointer chase in 4 MB"),
    "l2_bandwidth": ("L2", "read bandwidth, 8 MB working set"),
    "dram_latency": ("DRAM", "load latency, pointer chase in 256 MB"),
}


def parse_bench(text):
    return [{"metric": r["metric"], "unit": r["unit"], "value": float(r["value"]),
             "sm_mhz": float(r["sm_mhz"]) if r["sm_mhz"] else None}
            for r in csv.DictReader(io.StringIO(text))]


def measure(args):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True,
                         capture_output=True, text=True).stdout.splitlines()[0].strip()
    stamp = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": args.commit, "gpu": gpu}
    rows = []
    for run in range(1, args.repeats + 1):
        done = subprocess.run([str(args.bench)], check=True, capture_output=True, text=True)
        rows += [{**r, "run": run, **stamp} for r in parse_bench(done.stdout)]
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**r, "sm_mhz": "" if r["sm_mhz"] is None else r["sm_mhz"]} for r in rows)


def read_rows(path):
    with open(path, newline="") as f:
        return [{**r, "run": int(r["run"]), "value": float(r["value"]),
                 "sm_mhz": float(r["sm_mhz"]) if r["sm_mhz"] else None} for r in csv.DictReader(f)]


def grouped(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["metric"], r["unit"]), []).append(r)
    return groups


def number(value):
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def table_markdown(rows):
    runs = max(r["run"] for r in rows)
    first = rows[0]
    lines = [f"Measured {first['date']} at commit {first['commit']} on the {first['gpu']}: {runs} runs of "
             "`scripts/onchip_bench.cu`, median (min to max). The SM clock is measured inside each kernel "
             "(`clock64()` against `%globaltimer`).", "",
             "| tier | measure | unit | median | min to max | SM clock MHz |", "|---|---|---|---|---|---|"]
    for (metric, unit), group in grouped(rows).items():
        tier, label = LABELS.get(metric, ("", metric))
        values = [r["value"] for r in group]
        clocks = [r["sm_mhz"] for r in group if r["sm_mhz"] is not None]
        clock = f"{statistics.median(clocks):,.0f}" if clocks else ""
        lines.append(f"| {tier} | {label} | {unit} | {number(statistics.median(values))} | "
                     f"{number(min(values))} to {number(max(values))} | {clock} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run scripts/onchip_bench.cu several times and tabulate L2, shared memory, register and DRAM latency numbers")
    ap.add_argument("--bench", type=Path, default=Path("build/onchip_bench"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--commit", default="?")
    ap.add_argument("--out-dir", type=Path, default=Path("results/onchip"))
    ap.add_argument("--from-csv", type=Path, help="redraw the table from a saved CSV instead of measuring")
    args = ap.parse_args(argv)

    rows = read_rows(args.from_csv) if args.from_csv else measure(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "onchip.csv", rows)
    (args.out_dir / "onchip.md").write_text(table_markdown(rows))
    print(table_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
