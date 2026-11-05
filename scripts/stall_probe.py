#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_report  # noqa: E402

SENTENCE = "The scheduler gives every running sequence one row before any prompt gets the rest of the step. "
COLUMNS = ["token_budget", "run", "decoding", "long_prompt_tokens", "long_ttft_ms", "gap_ms_p99", "gap_ms_max",
           "date", "commit"]


def prompts_with_a_long_one(short_prompts, count, repeats):
    shorts = json.loads(Path(short_prompts).read_text())["prompts"][:count - 1]
    long = {"name": "long", "messages": [{"role": "user", "content": SENTENCE * repeats}]}
    return {"prompts": shorts + [long]}


def engine_command(engine, model, prompts, count, rate, output_len, budget, out, trace):
    cmd = [str(engine), "bench", "--model", str(model), "--max-batch", str(count), "--graphs",
           "--prompts", str(prompts), "--num-requests", str(count), "--output-len", str(output_len),
           "--request-rate", str(rate), "--seed", "0", "--out", str(out), "--trace", str(trace)]
    return cmd + ["--token-budget", str(budget)] if budget else cmd


def stall_run(requests, steps):
    long = max(requests, key=lambda r: int(r["prompt_tokens"]))
    arrived = float(long["arrival_s"])
    decoding = sum(float(r["arrival_s"]) + float(r["ttft_s"]) <= arrived < float(r["arrival_s"]) + float(r["latency_s"])
                   for r in requests if r is not long)
    gaps = sorted(load_report.token_gaps(steps))
    return {"decoding": decoding, "long_prompt_tokens": int(long["prompt_tokens"]),
            "long_ttft_ms": float(long["ttft_s"]) * 1e3, "gap_ms_p99": load_report.nearest_rank(gaps, 0.99) * 1e3,
            "gap_ms_max": gaps[-1] * 1e3}


def table_markdown(rows):
    budgets = sorted({int(r["token_budget"]) for r in rows})
    first = rows[0]
    lines = [f"Measured {first['date']} at commit {first['commit']}: a {first['long_prompt_tokens']}-token prompt "
             "arrives among short requests, medians of the runs.", "",
             "| token budget | long prompt tokens | decoding when it arrives | TTFT of the long prompt ms | gap p99 ms "
             "| largest gap ms | runs |", "|---|---|---|---|---|---|---|"]
    for budget in budgets:
        mine = [r for r in rows if int(r["token_budget"]) == budget]
        cells = [f"{statistics.median(float(r[k]) for r in mine):.0f}"
                 for k in ("long_prompt_tokens", "decoding", "long_ttft_ms", "gap_ms_p99", "gap_ms_max")]
        lines.append(f"| {budget or 'none'} | " + " | ".join(cells) + f" | {len(mine)} |")
    return "\n".join(lines) + "\n"


def measure(args):
    stamp = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": args.commit}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        prompts = Path(tmp) / "prompts.json"
        prompts.write_text(json.dumps(prompts_with_a_long_one(args.short_prompts, args.requests, args.repeat_sentence)))
        out, trace = Path(tmp) / "run.csv", Path(tmp) / "trace.json"
        for run in range(1, args.repeats + 1):
            for budget in args.budgets:
                subprocess.run(engine_command(args.engine, args.model, prompts, args.requests, args.rate,
                                              args.output_len, budget, out, trace),
                               check=True, capture_output=True, text=True)
                with open(out, newline="") as f:
                    requests = list(csv.DictReader(f))
                result = stall_run(requests, json.loads(trace.read_text())["steps"])
                rows.append({"token_budget": budget, "run": run, **result, **stamp})
                print(f"budget {budget} run {run}: {result}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode stalls when one long prompt arrives among short decoding requests")
    ap.add_argument("--engine", type=Path, default=Path("build/release/bin/engine"))
    ap.add_argument("--model", type=Path, default=Path("models/tinyllama"))
    ap.add_argument("--short-prompts", type=Path, default=Path("scripts/prompts.json"))
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--repeat-sentence", type=int, default=100)
    ap.add_argument("--rate", type=float, default=16.0)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 512, 2048])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--commit", default="?")
    ap.add_argument("--out-dir", type=Path, default=Path("results/load"))
    args = ap.parse_args(argv)

    rows = measure(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "stall.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "stall.md").write_text(table_markdown(rows))
    print(table_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
