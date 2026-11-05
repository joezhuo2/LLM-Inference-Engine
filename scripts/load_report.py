#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot  # noqa: E402

SYSTEMS = {"batched_graphs": "engine + graphs", "batched": "engine", "vllm_online": "vLLM",
           "sglang_online": "SGLang"}
ORDER = {"engine + graphs": 0, "engine": 1, "vLLM": 2, "SGLang": 3}
METRICS = ("ttft_s_p50", "ttft_s_p99", "tpot_s_p50", "tpot_s_p99", "output_throughput")
GAP_FIELDS = ("system", "rate", "run", "gaps", "gap_ms_p50", "gap_ms_p99", "gap_ms_max")
GAP_METRICS = ("gap_ms_p50", "gap_ms_p99", "gap_ms_max")


def system_of(row):
    system = SYSTEMS.get(row["driver"], row["driver"])
    budget = number(row.get("token_budget"))
    return f"{system}, budget {budget:.0f}" if budget > 0 else system


def series_order(system):
    base, _, budget = system.partition(", budget ")
    return ORDER.get(base, 9), int(budget or 0)


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def meets_slo(request, ttft_ms, tpot_ms):
    tpot = number(request["tpot_s"])
    return (number(request["ttft_s"]) * 1e3 <= ttft_ms
            and (math.isnan(tpot) or tpot * 1e3 <= tpot_ms))


def load_run(summary_path, ttft_ms, tpot_ms):
    with open(summary_path, newline="") as f:
        row = next(csv.DictReader(f))
    requests_path = summary_path.with_name(summary_path.name.replace("_summary.csv", ".csv"))
    with open(requests_path, newline="") as f:
        requests = list(csv.DictReader(f))
    met = sum(meets_slo(r, ttft_ms, tpot_ms) for r in requests)
    wall = number(row["wall_s"])
    run = {"system": system_of(row), "rate": number(row["request_rate"]),
           "attainment": met / len(requests), "goodput": met / wall,
           "preemptions": number(row.get("preemptions"))}
    run.update({key: number(row[key]) for key in METRICS})
    return run


def token_gaps(steps):
    last = {}
    gaps = []
    for step in steps:
        for seq in step["seqs"]:
            if seq in last:
                gaps.append(step["end_s"] - last[seq])
            last[seq] = step["end_s"]
    return gaps


def nearest_rank(ordered, q):
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def gap_run(summary_path, trace_path):
    with open(summary_path, newline="") as f:
        row = next(csv.DictReader(f))
    gaps = sorted(token_gaps(json.loads(trace_path.read_text())["steps"]))
    return {"system": system_of(row), "rate": number(row["request_rate"]),
            "run": int(re.search(r"_r(\d+)_summary", summary_path.name).group(1)), "gaps": len(gaps),
            "gap_ms_p50": statistics.median(gaps) * 1e3, "gap_ms_p99": nearest_rank(gaps, 0.99) * 1e3,
            "gap_ms_max": gaps[-1] * 1e3}


def write_gaps(path, runs):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_FIELDS, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow({**run, **{k: f"{run[k]:.3f}" for k in GAP_METRICS},
                             "rate": f"{run['rate']:g}"})


def read_gaps(path):
    with open(path, newline="") as f:
        return [{**r, "rate": float(r["rate"]), **{k: float(r[k]) for k in GAP_METRICS}}
                for r in csv.DictReader(f)]


def aggregate(runs, keys=(*METRICS, "attainment", "goodput", "preemptions")):
    groups = {}
    for run in runs:
        groups.setdefault((run["system"], run["rate"]), []).append(run)
    points = []
    for (system, rate), members in sorted(groups.items(),
                                          key=lambda g: (series_order(g[0][0]), g[0][1])):
        point = {"system": system, "rate": rate, "runs": len(members)}
        for key in keys:
            values = [m[key] for m in members if not math.isnan(m[key])]
            point[key] = statistics.median(values) if values else math.nan
        points.append(point)
    return points


def span(dates):
    first, last = min(dates)[:10], max(dates)[:10]
    return first if first == last else f"{first} to {last}"


def measured_line(summaries, commit):
    rows = []
    for path in summaries:
        with open(path, newline="") as f:
            rows.append(next(csv.DictReader(f)))
    engine = [r for r in rows if r["driver"].startswith("batched")]
    others = [r for r in rows if not r["driver"].startswith("batched")]
    parts = []
    if engine:
        parts.append(f"Engine runs measured {span([r['date'] for r in engine])} at commit {commit} "
                     f"on {engine[0]['gpu']}")
    for driver in sorted({r["driver"] for r in others}):
        mine = [r for r in others if r["driver"] == driver]
        parts.append(f"{system_of(mine[0])} runs measured {span([r['date'] for r in mine])} "
                     f"({mine[0]['version']})")
    return "; ".join(parts) + "."


def table_markdown(points, ttft_ms, tpot_ms):
    lines = [f"SLO: TTFT at most {ttft_ms:g} ms and TPOT at most {tpot_ms:g} ms; goodput is the "
             "requests per second that meet both.", "",
             "| system | QPS | output tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms "
             "| SLO met | goodput req/s | preemptions | runs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        preemptions = "" if math.isnan(p["preemptions"]) else f"{p['preemptions']:.0f}"
        lines.append(
            f"| {p['system']} | {p['rate']:g} | {p['output_throughput']:.0f} "
            f"| {p['ttft_s_p50'] * 1e3:.0f} | {p['ttft_s_p99'] * 1e3:.0f} "
            f"| {p['tpot_s_p50'] * 1e3:.1f} | {p['tpot_s_p99'] * 1e3:.1f} "
            f"| {p['attainment'] * 100:.0f}% | {p['goodput']:.2f} | {preemptions} | {p['runs']} |")
    return "\n".join(lines) + "\n"


def gaps_markdown(points):
    lines = ["Gaps between consecutive output tokens of one request, from the engine's step traces: every "
             "token of every request, medians of the runs' p50, p99 and largest gap.", "",
             "| system | QPS | gap p50 ms | gap p99 ms | largest gap ms | runs |", "|---|---|---|---|---|---|"]
    for p in points:
        lines.append(f"| {p['system']} | {p['rate']:g} | {p['gap_ms_p50']:.1f} | {p['gap_ms_p99']:.1f} "
                     f"| {p['gap_ms_max']:.0f} | {p['runs']} |")
    return "\n".join(lines) + "\n"


def curve_plot(points, key, scale, ylabel, title, out, theme, log=False, limit=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = plot._style(plt, theme)
    palette = list(t["series"].values())
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    for i, system in enumerate(sorted({p["system"] for p in points}, key=series_order)):
        members = [p for p in points if p["system"] == system]
        color = palette[i % len(palette)]
        ax.plot([p["rate"] for p in members], [p[key] * scale for p in members], color=color,
                linewidth=2, marker="o", markersize=8, markeredgecolor=t["surface"],
                markeredgewidth=2, label=system)
    if limit is not None:
        ax.axhline(limit, color=t["muted"], linewidth=1, linestyle="--")
        ax.text(ax.get_xlim()[0], limit, f" SLO {limit:g} ms", color=t["muted"], fontsize=9,
                va="bottom")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("offered load (requests per second)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", labelcolor=t["text"])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Report the QPS sweep in results/load")
    ap.add_argument("--dir", type=Path, default=Path("results/load"))
    ap.add_argument("--ttft-ms", type=float, default=500.0)
    ap.add_argument("--tpot-ms", type=float, default=50.0)
    ap.add_argument("--theme", choices=sorted(plot.THEMES), default="light")
    ap.add_argument("--commit", default="?", help="the engine commit the runs were measured at")
    ap.add_argument("--traces", type=Path,
                    help="step traces of the engine runs (<run>.json), summarized into token_gaps.csv")
    args = ap.parse_args(argv)

    summaries = sorted(args.dir.glob("*_summary.csv"))
    if not summaries:
        raise SystemExit(f"no summaries in {args.dir}")
    gaps_path = args.dir / "token_gaps.csv"
    if args.traces:
        traced = [(p, args.traces / p.name.replace("_summary.csv", ".json")) for p in summaries]
        write_gaps(gaps_path, [gap_run(p, t) for p, t in traced if t.exists()])
    points = aggregate(load_run(p, args.ttft_ms, args.tpot_ms) for p in summaries)
    table = measured_line(summaries, args.commit) + "\n\n" + table_markdown(points, args.ttft_ms, args.tpot_ms)
    if gaps_path.exists():
        gap_points = aggregate(read_gaps(gaps_path), GAP_METRICS)
        table += "\n" + gaps_markdown(gap_points)
        curve_plot(gap_points, "gap_ms_max", 1.0, "largest gap between two tokens (ms)",
                   "Longest decode stall against load", args.dir / "gap_max_vs_qps.png", args.theme,
                   log=True)
    (args.dir / "load_table.md").write_text(table)
    curve_plot(points, "ttft_s_p99", 1e3, "TTFT p99 (ms)", "Tail time to first token against load",
               args.dir / "ttft_p99_vs_qps.png", args.theme, log=True, limit=args.ttft_ms)
    curve_plot(points, "tpot_s_p99", 1e3, "TPOT p99 (ms)", "Tail time per output token against load",
               args.dir / "tpot_p99_vs_qps.png", args.theme, limit=args.tpot_ms)
    curve_plot(points, "goodput", 1.0, "requests per second meeting the SLO",
               "Goodput against load", args.dir / "goodput_vs_qps.png", args.theme)
    print(f"wrote {args.dir}/load_table.md and {4 if gaps_path.exists() else 3} plots from {len(summaries)} summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
