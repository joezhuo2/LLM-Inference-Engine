#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

NAME = re.compile(r"(on|off)_p(\d+)_q([\d.]+)_r(\d+)_summary\.csv")


def load(directory, traces):
    runs = []
    for path in sorted(Path(directory).glob("*_summary.csv")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        with open(path, newline="") as f:
            summary = next(csv.DictReader(f))
        with open(path.with_name(path.name.replace("_summary", "")), newline="") as f:
            ttft = {int(r["id"]): float(r["ttft_s"]) for r in csv.DictReader(f)}
        prefill = []
        trace = Path(traces) / path.name.replace("_summary.csv", ".json")
        if trace.exists():
            steps = json.loads(trace.read_text())["steps"]
            prefill = [s["end_s"] - s["start_s"] for s in steps
                       if s["kind"] == "prefill" and 0 not in s["seqs"]]
        runs.append({"cache": match[1], "prefix": int(match[2]), "rate": float(match[3]),
                     "own": summary["prompt_len"], "first_ttft": ttft[0],
                     "later_ttft": statistics.median(v for k, v in ttft.items() if k != 0),
                     "prefill": statistics.median(prefill) if prefill else None,
                     "hit_rate": float(summary["prefix_hit_rate"]),
                     "throughput": float(summary["output_throughput"]),
                     "date": summary["date"], "gpu": summary["gpu"]})
    return runs


def rows(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["prefix"], -run["rate"], run["cache"] == "off"), []).append(run)
    out = []
    for key in sorted(groups):
        group = groups[key]
        prefill = [r["prefill"] for r in group if r["prefill"] is not None]
        out.append({"prefix": group[0]["prefix"], "own": group[0]["own"], "rate": group[0]["rate"],
                    "cache": group[0]["cache"], "runs": len(group),
                    "first_ttft": statistics.median(r["first_ttft"] for r in group),
                    "later_ttft": statistics.median(r["later_ttft"] for r in group),
                    "prefill": statistics.median(prefill) if prefill else None,
                    "hit_rate": statistics.median(r["hit_rate"] for r in group),
                    "throughput": statistics.median(r["throughput"] for r in group)})
    return out


def table(rows):
    lines = ["| shared prefix | own tokens | arrivals | cache | first request TTFT ms | "
             "later requests TTFT p50 ms | prefill step p50 ms | hit rate | output tok/s | runs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        arrivals = "at once" if r["rate"] == 0 else f"{r['rate']:g}/s"
        prefill = "" if r["prefill"] is None else f"{r['prefill'] * 1e3:.1f}"
        lines.append(f"| {r['prefix']} | {r['own']} | {arrivals} | {r['cache']} | "
                     f"{r['first_ttft'] * 1e3:.1f} | {r['later_ttft'] * 1e3:.1f} | {prefill} | "
                     f"{100 * r['hit_rate']:.1f}% | {r['throughput']:.1f} | {r['runs']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Tabulate scripts/prefix_cache_sweep.sh: medians of "
                                             "the runs per prefix, arrival rate and cache setting")
    ap.add_argument("--dir", type=Path, default=Path("results/prefix"))
    ap.add_argument("--traces", type=Path, default=Path("build/prefix_traces"))
    args = ap.parse_args()
    runs = load(args.dir, args.traces)
    if not runs:
        raise SystemExit(f"no prefix cache runs in {args.dir}")
    print(table(rows(runs)))
    print()
    print(f"{runs[0]['gpu']}, {min(r['date'] for r in runs)[:10]}")


if __name__ == "__main__":
    main()
