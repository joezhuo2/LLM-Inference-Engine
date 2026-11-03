#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot  # noqa: E402

SYSTEMS = {"batched_graphs": "engine + graphs", "batched": "engine", "vllm_online": "vLLM",
           "sglang_online": "SGLang"}
COLORS = {"engine + graphs": 0, "engine": 1, "vLLM": 2, "SGLang": 3}
METRICS = ("ttft_s_p50", "ttft_s_p99", "tpot_s_p50", "tpot_s_p99", "output_throughput")


def system_of(driver):
    return SYSTEMS.get(driver, driver)


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
    run = {"system": system_of(row["driver"]), "rate": number(row["request_rate"]),
           "attainment": met / len(requests), "goodput": met / wall,
           "preemptions": number(row.get("preemptions"))}
    run.update({key: number(row[key]) for key in METRICS})
    return run


def aggregate(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["system"], run["rate"]), []).append(run)
    points = []
    for (system, rate), members in sorted(groups.items()):
        point = {"system": system, "rate": rate, "runs": len(members)}
        for key in (*METRICS, "attainment", "goodput", "preemptions"):
            values = [m[key] for m in members if not math.isnan(m[key])]
            point[key] = statistics.median(values) if values else math.nan
        points.append(point)
    return points


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


def curve_plot(points, key, scale, ylabel, title, out, theme, log=False, limit=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = plot._style(plt, theme)
    palette = list(t["series"].values())
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    for system in sorted({p["system"] for p in points}, key=lambda s: COLORS.get(s, 9)):
        members = [p for p in points if p["system"] == system]
        color = palette[COLORS.get(system, 0) % len(palette)]
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
    args = ap.parse_args(argv)

    summaries = sorted(args.dir.glob("*_summary.csv"))
    if not summaries:
        raise SystemExit(f"no summaries in {args.dir}")
    points = aggregate(load_run(p, args.ttft_ms, args.tpot_ms) for p in summaries)
    (args.dir / "load_table.md").write_text(table_markdown(points, args.ttft_ms, args.tpot_ms))
    curve_plot(points, "ttft_s_p99", 1e3, "TTFT p99 (ms)", "Tail time to first token against load",
               args.dir / "ttft_p99_vs_qps.png", args.theme, log=True, limit=args.ttft_ms)
    curve_plot(points, "tpot_s_p99", 1e3, "TPOT p99 (ms)", "Tail time per output token against load",
               args.dir / "tpot_p99_vs_qps.png", args.theme, limit=args.tpot_ms)
    curve_plot(points, "goodput", 1.0, "requests per second meeting the SLO",
               "Goodput against load", args.dir / "goodput_vs_qps.png", args.theme)
    print(f"wrote {args.dir}/load_table.md and three plots from {len(summaries)} summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
