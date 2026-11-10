#!/usr/bin/env python3
import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_llamacpp as bl  # noqa: E402
import plot  # noqa: E402

ENGINE = "engine"
GRAPHS = "engine + graphs"
PAGED = "engine, paged prefill"
LLAMA = "llama.cpp -ub 512"
LLAMA_FULL = "llama.cpp -ub 2048"
SYSTEMS = [PAGED, ENGINE, GRAPHS, LLAMA, LLAMA_FULL]
COLORS = {PAGED: "engine", ENGINE: "engine", GRAPHS: "engine + graphs", LLAMA: "llama.cpp", LLAMA_FULL: "HF transformers"}
COLUMNS = ["system", "prompt_tokens", "run", "ttft_s", "date", "commit"]


def engine_command(engine, model, prompt_len, run, warmup, out, graphs, budget=0, paged=False):
    cmd = [str(engine), "bench", "--model", str(model), "--max-batch", "1", "--num-requests", "1",
           "--prompt-len", str(prompt_len), "--output-len", "1", "--warmup", str(warmup),
           "--seed", str(run), "--out", str(out)]
    if graphs:
        cmd.append("--graphs")
    if paged:
        cmd += ["--prefill-attention", "paged"]
    return cmd + ["--token-budget", str(budget)] if budget else cmd


def base_system(system):
    return system.partition(", budget ")[0]


def llama_command(binary, model, lengths, ubatch):
    longest = max(lengths)
    return [str(binary), "-m", str(model), "-npp", ",".join(str(n) for n in lengths), "-ntg", "1",
            "-npl", "1,1", "-c", str(longest + 1), "-b", str(longest), "-ub", str(ubatch),
            "-ngl", "99", "--output-format", "jsonl"]


def read_engine_ttft(summary):
    with open(summary, newline="") as f:
        return float(next(csv.DictReader(f))["ttft_s_p50"])


def llama_ttfts(rows):
    seen = set()
    ttfts = {}
    for row in rows:
        if row["pp"] in seen:
            ttfts[row["pp"]] = float(row["t_pp"])
        seen.add(row["pp"])
    return ttfts


def measure(args):
    stamp = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": args.commit}
    measured = []
    engines = [(PAGED, False, 0, True), (ENGINE, False, 0, False), (GRAPHS, True, 0, False)]
    if args.token_budget:
        engines.append((f"{GRAPHS}, budget {args.token_budget}", True, args.token_budget, False))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bench.csv"
        for run in range(1, args.repeats + 1):
            for length in args.lengths:
                for system, graphs, budget, paged in engines:
                    command = engine_command(args.engine, args.model, length, run, args.warmup, out, graphs, budget, paged)
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    measured.append({"system": system, "prompt_tokens": length, "run": run,
                                     "ttft_s": read_engine_ttft(Path(tmp) / "bench_summary.csv"), **stamp})
            if not args.llama_batched_bench:
                continue
            for system, ubatch in ((LLAMA, 512), (LLAMA_FULL, max(args.lengths))):
                done = subprocess.run(llama_command(args.llama_batched_bench, args.gguf, args.lengths, ubatch),
                                      check=True, capture_output=True, text=True)
                for length, ttft in llama_ttfts(bl.parse_batched_bench(done.stdout)).items():
                    measured.append({"system": system, "prompt_tokens": length, "run": run, "ttft_s": ttft, **stamp})
    return measured


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="") as f:
        return [{**r, "prompt_tokens": int(r["prompt_tokens"]), "run": int(r["run"]), "ttft_s": float(r["ttft_s"])}
                for r in csv.DictReader(f)]


def grouped(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["system"], {}).setdefault(r["prompt_tokens"], []).append(r["ttft_s"])
    return {s: groups[s] for s in sorted(groups, key=lambda s: (SYSTEMS.index(base_system(s)), s))}


def table_markdown(rows):
    groups = grouped(rows)
    lengths = sorted({r["prompt_tokens"] for r in rows})
    runs = max(r["run"] for r in rows)
    header = "| system | " + " | ".join(f"{n:,}" for n in lengths) + " |"
    rule = "|---|" + "---|" * len(lengths)
    lines = [f"Measured {rows[0]['date']} at commit {rows[0]['commit']}: one request per run, median of {runs} runs.",
             "", "TTFT in ms against prompt tokens, median (min to max):", "", header, rule]
    for system, by_length in groups.items():
        cells = [f"{statistics.median(by_length[n]) * 1e3:.1f} ({min(by_length[n]) * 1e3:.1f} to "
                 f"{max(by_length[n]) * 1e3:.1f})" for n in lengths]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += ["", "Median TTFT in ms per prompt token:", "", header, rule]
    for system, by_length in groups.items():
        cells = [f"{statistics.median(by_length[n]) * 1e3 / n:.3f}" for n in lengths]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_ttft(rows, out, theme="light"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = plot._style(plt, theme)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    for system, by_length in grouped(rows).items():
        xs = sorted(by_length)
        base = base_system(system)
        ax.plot(xs, [statistics.median(by_length[n]) * 1e3 for n in xs], color=t["series"][COLORS[base]],
                linestyle="-" if system == base and system != PAGED else "--", linewidth=2, marker="o", markersize=8,
                markeredgecolor=t["surface"], markeredgewidth=2, label=system)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("single-request TTFT (ms)")
    ax.set_title("Prefill time against prompt length")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best", labelcolor=t["text"])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single-request TTFT against prompt length, engine and llama.cpp")
    ap.add_argument("--engine", type=Path, default=Path("build/release/bin/engine"))
    ap.add_argument("--model", type=Path, default=Path("models/tinyllama"))
    ap.add_argument("--gguf", type=Path, default=Path("models/tinyllama-bf16.gguf"))
    ap.add_argument("--llama-batched-bench", type=Path, help="llama-batched-bench binary; leave out to skip llama.cpp")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--token-budget", type=int, default=0, help="also time the engine with graphs and this budget")
    ap.add_argument("--commit", default="?")
    ap.add_argument("--out-dir", type=Path, default=Path("results/prefill"))
    ap.add_argument("--from-csv", type=Path, help="redraw the table and plot from a saved CSV instead of measuring")
    args = ap.parse_args(argv)

    if args.from_csv:
        rows = read_rows(args.from_csv)
    else:
        rows = measure(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "prefill_ttft.csv", rows)
    (args.out_dir / "prefill.md").write_text(table_markdown(rows))
    plot_ttft(rows, args.out_dir / "prefill_ttft.png")
    print(table_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
