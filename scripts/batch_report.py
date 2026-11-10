#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path

METRICS = ("tpot_s_p50", "output_throughput", "ttft_s_p50", "steps", "peak_kv_blocks")


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load(directory):
    runs = []
    for path in sorted(Path(directory).glob("*_summary.csv")):
        with open(path, newline="") as f:
            row = next(csv.DictReader(f))
        runs.append({"name": path.name, "graphs": row["driver"] == "batched_graphs",
                     "max_batch": int(row["max_batch"]), "requests": int(row["num_requests"]),
                     "graph_steps": int(row.get("graph_steps") or 0),
                     "date": row["date"], "gpu": row["gpu"],
                     **{key: number(row.get(key)) for key in METRICS}})
    return runs


def median_by(runs, key):
    groups = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return {k: {"runs": len(v), **{m: statistics.median(r[m] for r in v) for m in METRICS}}
            for k, v in groups.items()}


def batch_rows(runs):
    sweep = [r for r in runs if not r["name"].startswith("pad_")]
    medians = median_by(sweep, lambda r: (r["max_batch"], r["graphs"]))
    rows = []
    for batch in sorted({b for b, _ in medians}):
        eager, graphs = medians.get((batch, False)), medians.get((batch, True))
        rows.append({"max_batch": batch, "eager": eager, "graphs": graphs})
    return rows


def pad_rows(runs):
    pad = [r for r in runs if r["name"].startswith("pad_")]
    medians = median_by(pad, lambda r: r["graph_steps"])
    return [{"graph_steps": s, **medians[s]} for s in sorted(medians)]


def ms(value):
    return "" if math.isnan(value) else f"{value * 1e3:.2f}"


def whole(value):
    return "" if math.isnan(value) else f"{value:.0f}"


def markdown(runs):
    nan = {m: math.nan for m in METRICS}
    lines = ["| max batch | eager TPOT p50 (ms) | graphs TPOT p50 (ms) | eager tokens/s | "
             "graphs tokens/s | peak KV blocks |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in batch_rows(runs):
        e, g = row["eager"] or nan, row["graphs"] or nan
        lines.append(f"| {row['max_batch']} | {ms(e['tpot_s_p50'])} | {ms(g['tpot_s_p50'])} | "
                     f"{whole(e['output_throughput'])} | {whole(g['output_throughput'])} | "
                     f"{whole(max(e['peak_kv_blocks'], g['peak_kv_blocks']))} |")
    pads = pad_rows(runs)
    if pads:
        lines += ["", "| graph steps | TPOT p50 (ms) | tokens/s |", "|---:|---:|---:|"]
        for row in pads:
            lines.append(f"| {row['graph_steps']} | {ms(row['tpot_s_p50'])} | "
                         f"{whole(row['output_throughput'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Median tables of scripts/batch_sweep.sh runs")
    parser.add_argument("directory", nargs="?", default="results/batch")
    parser.add_argument("--out", help="markdown file to write as well as printing it")
    args = parser.parse_args()
    runs = load(args.directory)
    if not runs:
        raise SystemExit(f"no *_summary.csv in {args.directory}")
    text = markdown(runs)
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
