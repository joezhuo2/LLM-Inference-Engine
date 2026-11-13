#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUNNING = re.compile(r"running_(repeat|chat)_n(\d+)_(k\d+a?)_r(\d+)_summary\.csv")
ONE = re.compile(r"(sampled|draft)_(repeat|chat)_(.+)_([kd]\d+a?)_r(\d+)_summary\.csv")
LOAD = re.compile(r"load_chat_q([\d.]+)_(k\d+a?)_r(\d+)_summary\.csv")
BASE = re.compile(r"(repeat|chat)_(.+)_k(\d+)_r(\d+)_summary\.csv")
LABEL = {"k0": "none", "k8": "fixed", "k8a": "adaptive"}


def row_of(path):
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def median(values):
    return statistics.median(values)


def decode_s(row):
    return float(row["tpot_s_p50"]) * (int(row["output_tokens"]) - 1)


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def running(directory):
    runs = defaultdict(list)
    for path in sorted(directory.glob("running_*_summary.csv")):
        m = RUNNING.fullmatch(path.name)
        if m:
            runs[(m[1], int(m[2]), m[3])].append(row_of(path))
    rows = []
    for (name, n, k), group in sorted(runs.items(), key=lambda i: (i[0][0] != "repeat", i[0][1], i[0][2])):
        rate = median(float(r["output_throughput"]) for r in group)
        plain = runs.get((name, n, "k0"))
        base = median(float(r["output_throughput"]) for r in plain) if plain else rate
        first = group[0]
        drafted, accepted = int(first["drafted"]), int(first["draft_accepted"])
        rows.append([name, n, LABEL[k], first["output_tokens"], first["steps"], drafted, accepted,
                     f"{100 * accepted / drafted:.1f}" if drafted else "0.0", f"{rate:.0f}",
                     f"{rate / base:.2f}", len(group)])
    table(["prompts", "requests", "guesses", "output tokens", "steps", "draft tokens", "accepted",
           "accepted %", "tokens/s", "against none", "runs"], rows)


def one_at_a_time(directory, part, baseline):
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.glob(f"{part}_*_summary.csv")):
        m = ONE.fullmatch(path.name)
        if m and m[1] == part:
            runs[(m[2], m[4])][int(m[5])].append(row_of(path))
    if baseline:
        for path in sorted(baseline.glob("*_summary.csv")):
            m = BASE.fullmatch(path.name)
            if m and m[3] in ("0", "8"):
                runs[(m[1], "k" + m[3])][int(m[4])].append(row_of(path))
    rows = []
    plain_ms = {}
    order = {"k0": 0, "k8": 1, "k8a": 2}
    for (name, k), by_run in sorted(runs.items(), key=lambda i: (i[0][0] != "repeat", order.get(i[0][1], 3), i[0][1])):
        totals = [(sum(int(r["output_tokens"]) for r in group), sum(decode_s(r) for r in group),
                   sum(int(r["steps"]) for r in group), sum(int(r["drafted"]) for r in group),
                   sum(int(r["draft_accepted"]) for r in group)) for group in by_run.values()]
        tokens, _, steps, drafted, accepted = totals[0]
        prompts = len(next(iter(by_run.values())))
        ms = median(1e3 * t[1] / (t[0] - prompts) for t in totals)
        if k == "k0":
            plain_ms[name] = ms
        label = LABEL.get(k, f"draft model, up to {k[1:]}")
        if baseline and k == "k8":
            label = "prompt lookup, up to 8"
        rows.append([name, label, tokens, steps, drafted, accepted,
                     f"{100 * accepted / drafted:.1f}" if drafted else "0.0", f"{ms:.2f}",
                     f"{plain_ms[name] / ms:.2f}" if name in plain_ms else "", len(by_run)])
    table(["prompts", "guesses", "output tokens", "steps", "draft tokens", "accepted", "accepted %",
           "ms per token", "speedup", "runs"], rows)


def load(directory):
    runs = defaultdict(list)
    for path in sorted(directory.glob("load_*_summary.csv")):
        m = LOAD.fullmatch(path.name)
        if m:
            runs[(float(m[1]), m[2])].append(row_of(path))
    rows = []
    for (rate, k), group in sorted(runs.items()):
        first = group[0]
        ms = lambda field: f"{1e3 * median(float(r[field]) for r in group):.1f}"
        rows.append([f"{rate:g}", LABEL[k], first["requests"], first["steps"], first["drafted"],
                     first["draft_accepted"], ms("ttft_s_p50"), ms("tpot_s_p50"), ms("tpot_s_p99"),
                     ms("latency_s_p50"), len(group)])
    table(["requests/s", "guesses", "requests", "steps", "draft tokens", "accepted", "TTFT p50 ms",
           "TPOT p50 ms", "TPOT p99 ms", "latency p50 ms", "runs"], rows)


def main():
    parser = argparse.ArgumentParser(description="Tables of scripts/speculative_more.sh")
    parser.add_argument("--dir", type=Path, default=Path("results/speculative_more"))
    parser.add_argument("--baseline", type=Path, default=Path("results/speculative"))
    args = parser.parse_args()
    some = row_of(next(iter(sorted(args.dir.glob("*_summary.csv")))))
    print(f"{some['date'][:10]}, {some['gpu']}\n")
    print("Several requests decoding at once:\n")
    running(args.dir)
    print("Sampled instead of greedy, one request at a time:\n")
    one_at_a_time(args.dir, "sampled", None)
    print("A draft model, one request at a time, greedy:\n")
    one_at_a_time(args.dir, "draft", args.baseline)
    print("Under load, chat prompts:\n")
    load(args.dir)


if __name__ == "__main__":
    main()
