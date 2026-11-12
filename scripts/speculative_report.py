#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

NAME = re.compile(r"(repeat|chat)_(.+)_k(\d+)_r(\d+)_summary\.csv")


def load(directory):
    runs = []
    for path in sorted(Path(directory).glob("*_summary.csv")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        with open(path, newline="") as f:
            row = next(csv.DictReader(f))
        tokens = int(row["output_tokens"])
        runs.append({"set": match[1], "prompt": match[2], "k": int(match[3]), "run": int(match[4]),
                     "tokens": tokens, "steps": int(row["steps"]), "drafted": int(row["drafted"]),
                     "accepted": int(row["draft_accepted"]),
                     "decode_s": float(row["tpot_s_p50"]) * (tokens - 1),
                     "date": row["date"], "gpu": row["gpu"]})
    return runs


def trace_tokens(traces, name, prompt, k):
    path = Path(traces) / f"{name}_{prompt}_k{k}.json"
    if not path.exists():
        return None
    return [t["id"] for step in json.loads(path.read_text())["steps"] for t in step["tokens"]]


def same_output(traces, name, prompts, k):
    same = 0
    for prompt in prompts:
        plain, spec = trace_tokens(traces, name, prompt, 0), trace_tokens(traces, name, prompt, k)
        if plain is None or spec is None:
            return ""
        same += plain == spec
    return f"{same} of {len(prompts)}"


def rows(runs, traces):
    out = []
    for name in ("repeat", "chat"):
        mine = [r for r in runs if r["set"] == name]
        base = None
        for k in sorted({r["k"] for r in mine}):
            at = [r for r in mine if r["k"] == k]
            run_ids = sorted({r["run"] for r in at})
            first = [r for r in at if r["run"] == run_ids[0]]
            tokens = sum(r["tokens"] for r in first)
            for run in run_ids:
                if sum(r["tokens"] for r in at if r["run"] == run) != tokens:
                    raise SystemExit(f"{name}, {k} drafts: run {run} wrote a different number of tokens")
            ms = statistics.median(
                1e3 * sum(r["decode_s"] for r in at if r["run"] == run) / (tokens - len(first))
                for run in run_ids)
            if base is None:
                base = ms
            drafted = sum(r["drafted"] for r in first)
            accepted = sum(r["accepted"] for r in first)
            steps = sum(r["steps"] for r in first)
            out.append({"set": name, "k": k, "prompts": len(first), "tokens": tokens, "steps": steps,
                        "drafted": drafted, "accepted": accepted,
                        "accepted_pct": 100 * accepted / drafted if drafted else 0.0,
                        "tokens_per_step": tokens / steps, "ms": ms, "speedup": base / ms,
                        "same": same_output(traces, name, [r["prompt"] for r in first], k),
                        "runs": len(run_ids)})
    return out


def prompt_rows(runs, k):
    out = []
    for name in ("repeat", "chat"):
        mine = [r for r in runs if r["set"] == name]
        for prompt in dict.fromkeys(r["prompt"] for r in mine):
            def ms(at):
                return statistics.median(1e3 * r["decode_s"] / (r["tokens"] - 1) for r in at)
            plain = [r for r in mine if r["prompt"] == prompt and r["k"] == 0]
            spec = [r for r in mine if r["prompt"] == prompt and r["k"] == k]
            if not plain or not spec or plain[0]["tokens"] < 2:
                continue
            out.append({"set": name, "prompt": prompt, "tokens": spec[0]["tokens"], "steps": spec[0]["steps"],
                        "drafted": spec[0]["drafted"], "accepted": spec[0]["accepted"],
                        "plain_ms": ms(plain), "ms": ms(spec)})
    return out


def table(rows):
    lines = ["| prompts | max draft | output tokens | steps | draft tokens | accepted | accepted % | tokens per step | ms per token | speedup | same output | runs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['set']} | {r['k']} | {r['tokens']} | {r['steps']} | {r['drafted']} | {r['accepted']} | "
                     f"{r['accepted_pct']:.1f} | {r['tokens_per_step']:.2f} | {r['ms']:.2f} | {r['speedup']:.2f} | "
                     f"{r['same']} | {r['runs']} |")
    return "\n".join(lines)


def prompt_table(rows, k):
    lines = [f"| prompts | prompt | output tokens | steps | draft tokens | accepted | ms per token, no drafts | ms per token, up to {k} | speedup |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['set']} | {r['prompt']} | {r['tokens']} | {r['steps']} | {r['drafted']} | {r['accepted']} | "
                     f"{r['plain_ms']:.2f} | {r['ms']:.2f} | {r['plain_ms'] / r['ms']:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Markdown tables of scripts/speculative_sweep.sh: per prompt set and draft length, the steps, the draft tokens and how many were accepted, the time per output token after the first (median of the runs) and for how many prompts the first run wrote the tokens of the run without drafts; then every prompt at one draft length")
    parser.add_argument("directory", nargs="?", default="results/speculative")
    parser.add_argument("--per-prompt", type=int, default=8, help="draft length of the per-prompt table")
    parser.add_argument("--traces", help="traces of the first runs (default: DIRECTORY/traces)")
    args = parser.parse_args()
    runs = load(args.directory)
    if not runs:
        raise SystemExit(f"no summary CSVs in {args.directory}")
    print(table(rows(runs, args.traces or Path(args.directory) / "traces")))
    print()
    print(prompt_table(prompt_rows(runs, args.per_prompt), args.per_prompt))
    print()
    print(f"{runs[0]['gpu']}, {min(r['date'] for r in runs)[:10]}")


if __name__ == "__main__":
    main()
