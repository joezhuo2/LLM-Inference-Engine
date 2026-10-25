#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

DECODE_WEIGHT_BYTES = 2_069_024_768
SYSTEMS = ["engine", "engine + graphs", "HF transformers", "llama.cpp"]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
        "series": {"engine": "#2a78d6", "engine + graphs": "#7b4fd1", "HF transformers": "#eb6834", "llama.cpp": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
        "series": {"engine": "#3987e5", "engine + graphs": "#9a78e6", "HF transformers": "#d95926", "llama.cpp": "#199e70"},
    },
}

NUMERIC = ("max_batch", "output_throughput", "tpot_s_p50", "tpot_s_p99", "ttft_s_p50", "ttft_s_p99",
           "bandwidth_gb_s", "requests", "output_tokens")


def system_of(driver: str) -> str:
    if driver.startswith("hf"):
        return "HF transformers"
    if driver.startswith("llamacpp"):
        return "llama.cpp"
    if driver == "batched_graphs":
        return "engine + graphs"
    return "engine"


def load_summaries(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                row = dict(raw)
                for key in NUMERIC:
                    try:
                        row[key] = float(raw[key])
                    except (KeyError, TypeError, ValueError):
                        row[key] = math.nan
                if math.isnan(row["bandwidth_gb_s"]) and row["max_batch"] == 1 and row["tpot_s_p50"] > 0:
                    row["bandwidth_gb_s"] = DECODE_WEIGHT_BYTES / row["tpot_s_p50"] / 1e9
                row["system"] = system_of(raw.get("driver", ""))
                row["source"] = str(path)
                rows.append(row)
    return rows


def aggregate(rows, key):
    groups = {}
    for r in rows:
        if math.isnan(r[key]) or math.isnan(r["max_batch"]):
            continue
        groups.setdefault((r["system"], int(r["max_batch"])), []).append(r[key])
    return {k: statistics.median(v) for k, v in groups.items()}


def series(rows, key):
    data = aggregate(rows, key)
    out = {}
    for (system, batch), value in sorted(data.items(), key=lambda kv: (SYSTEMS.index(kv[0][0]), kv[0][1])):
        out.setdefault(system, []).append((batch, value))
    return out


def table_markdown(rows):
    header = ["system", "max batch", "output tok/s", "TPOT p50 ms", "TTFT p50 ms", "weights GB/s", "runs"]
    keys = {}
    for r in rows:
        keys.setdefault((r["system"], int(r["max_batch"]) if not math.isnan(r["max_batch"]) else 0), []).append(r)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    def cell(values, scale=1.0, digits=1):
        vals = [v for v in values if not math.isnan(v)]
        return f"{statistics.median(vals) * scale:.{digits}f}" if vals else ""

    for (system, batch), group in sorted(keys.items(), key=lambda kv: (SYSTEMS.index(kv[0][0]), kv[0][1])):
        lines.append("| " + " | ".join([
            system, str(batch),
            cell([g["output_throughput"] for g in group]),
            cell([g["tpot_s_p50"] for g in group], 1e3, 2),
            cell([g["ttft_s_p50"] for g in group], 1e3, 1),
            cell([g["bandwidth_gb_s"] for g in group]),
            str(len(group)),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _style(plt, theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["muted"], "text.color": t["text"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    return t


def _place(members, ys, gap):
    center = sum(ys[i] for i in members) / len(members)
    start = center - gap * (len(members) - 1) / 2
    return [start + k * gap for k in range(len(members))]


def spread_labels(ys, gap):
    clusters = []
    for i in sorted(range(len(ys)), key=ys.__getitem__):
        clusters.append([i])
        while len(clusters) > 1 and _place(clusters[-2], ys, gap)[-1] + gap > _place(clusters[-1], ys, gap)[0]:
            clusters[-2:] = [clusters[-2] + clusters[-1]]
    out = list(ys)
    for members in clusters:
        for i, y in zip(members, _place(members, ys, gap)):
            out[i] = y
    return out


def line_plot(rows, key, scale, ylabel, title, out, theme, note=None):
    import matplotlib.transforms as mtransforms
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _style(plt, theme)
    data = series(rows, key)
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ends = []
    for system, points in data.items():
        xs = [p[0] for p in points]
        ys = [p[1] * scale for p in points]
        color = t["series"][system]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=t["surface"], markeredgewidth=2, label=system)
        ends.append((xs[-1], ys[-1]))
    batches = sorted({p[0] for pts in data.values() for p in pts})
    if len(batches) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlim(left=min(batches) * 0.8, right=max(batches) * 1.35)
    ax.set_ylim(bottom=0)
    gap = 0.055 * ax.get_ylim()[1]
    shifted = mtransforms.offset_copy(ax.transData, fig=fig, x=8, units="points")
    for x in {e[0] for e in ends}:
        at_x = [i for i, e in enumerate(ends) if e[0] == x]
        for i, y in zip(at_x, spread_labels([ends[i][1] for i in at_x], gap)):
            value = ends[i][1]
            ax.text(x, y, f"{value:.0f}" if scale == 1 and value >= 10 else f"{value:.1f}",
                    transform=shifted, va="center", color=t["text"], fontsize=10)
    ax.set_xlabel("max batch (concurrent requests)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", labelcolor=t["text"])
    if note:
        fig.text(0.01, 0.01, note, color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return True


def bandwidth_plot(rows, peak, target_pct, out, theme):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _style(plt, theme)
    data = {s: dict(pts) for s, pts in series(rows, "bandwidth_gb_s").items()}
    bars = [(s, d[1]) for s, d in data.items() if 1 in d]
    if not bars:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 0.9 * len(bars) + 2.4), dpi=160)
    names = [b[0] for b in bars]
    values = [b[1] for b in bars]
    ypos = list(range(len(bars)))[::-1]
    ax.barh(ypos, values, height=0.5, color=[t["series"][n] for n in names])
    for y, v in zip(ypos, values):
        ax.text(v + peak * 0.01, y, f"{v:.0f} GB/s ({100 * v / peak:.0f}%)", va="center",
                color=t["text"], fontsize=10)
    ax.axvline(peak, color=t["muted"], linewidth=1.5)
    ax.text(peak + peak * 0.01, len(bars) - 0.35, f"measured peak\n{peak:g} GB/s", ha="left",
            va="top", color=t["muted"], fontsize=9)
    target = peak * target_pct / 100.0
    ax.axvline(target, color=t["muted"], linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(target - peak * 0.01, len(bars) - 0.35, f"{target_pct:g}% target", ha="right", va="top",
            color=t["muted"], fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, peak * 1.4)
    ax.set_ylim(-0.6, len(bars) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("weights read per decode token / median TPOT (GB/s)")
    ax.set_title("Batch-1 decode bandwidth")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot engine and baseline summary CSVs")
    ap.add_argument("summaries", nargs="+", type=Path, help="*_summary.csv files")
    ap.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--peak-gb-s", type=float, default=323.6)
    ap.add_argument("--target-pct", type=float, default=60.0)
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = ap.parse_args(argv)

    rows = load_summaries(args.summaries)
    if not rows:
        ap.error("no rows in the given CSVs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpus = sorted({r.get("gpu", "") for r in rows if r.get("gpu")})
    note = f"{', '.join(gpus)}; medians over runs" if gpus else None

    written = []
    jobs = [
        ("throughput_vs_batch.png", lambda p: line_plot(
            rows, "output_throughput", 1.0, "output tokens per second", "Throughput vs concurrency", p,
            args.theme, note)),
        ("tpot_vs_batch.png", lambda p: line_plot(
            rows, "tpot_s_p50", 1e3, "median time per output token (ms)", "Decode latency vs concurrency",
            p, args.theme, note)),
        ("batch1_bandwidth.png", lambda p: bandwidth_plot(rows, args.peak_gb_s, args.target_pct, p, args.theme)),
    ]
    for name, job in jobs:
        path = args.out_dir / name
        if job(path):
            written.append(path)
    table = args.out_dir / "summary_table.md"
    table.write_text(table_markdown(rows))
    written.append(table)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
