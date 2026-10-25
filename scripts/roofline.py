#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot  # noqa: E402

PEAK_GB_S = 323.6
PEAK_TFLOPS = 34.5
BF16 = 2
OPS = ("qkv", "o_proj", "gate_up", "down", "lm_head", "attention")
OP_ROW = re.compile(r"^\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$")
STEP_SYSTEMS = ("engine + graphs", "engine", "llama.cpp", "HF transformers")
PLOT_SYSTEMS = ("engine + graphs", "llama.cpp", "HF transformers")


def model_shape(config):
    heads = config["num_attention_heads"]
    return {
        "hidden": config["hidden_size"],
        "intermediate": config["intermediate_size"],
        "layers": config["num_hidden_layers"],
        "heads": heads,
        "kv_heads": config.get("num_key_value_heads", heads),
        "head_dim": config["hidden_size"] // heads,
        "vocab": config["vocab_size"],
    }


def gemm_cost(rows, n_in, n_out, count):
    params = n_in * n_out
    return {
        "flops": 2.0 * rows * params * count,
        "bytes": float(BF16 * (params + rows * (n_in + n_out)) * count),
    }


def op_costs(shape, rows, context):
    h = shape["hidden"]
    kv = shape["kv_heads"] * shape["head_dim"]
    q = shape["heads"] * shape["head_dim"]
    layers = shape["layers"]
    return {
        "qkv": gemm_cost(rows, h, q + 2 * kv, layers),
        "o_proj": gemm_cost(rows, q, h, layers),
        "gate_up": gemm_cost(rows, h, 2 * shape["intermediate"], layers),
        "down": gemm_cost(rows, shape["intermediate"], h, layers),
        "lm_head": gemm_cost(rows, h, shape["vocab"], 1),
        "attention": {
            "flops": 4.0 * rows * q * context * layers,
            "bytes": float(BF16 * rows * (2 * kv * context + 2 * q) * layers),
        },
    }


def step_cost(shape, rows, context):
    costs = op_costs(shape, rows, context).values()
    return {"flops": sum(c["flops"] for c in costs), "bytes": sum(c["bytes"] for c in costs)}


def roof(intensity, peak_gb_s, peak_tflops):
    return min(peak_tflops * 1e12, intensity * peak_gb_s * 1e9)


def parse_op_table(text):
    times = {}
    in_table = False
    for line in text.splitlines():
        if line.lstrip("#").strip().startswith("GPU time per op"):
            in_table = True
            continue
        if in_table and line.startswith("#"):
            break
        m = OP_ROW.match(line) if in_table else None
        if m:
            times[m[1]] = float(m[3])
    return times


def point(name, rows, cost, seconds, peak_gb_s, peak_tflops):
    intensity = cost["flops"] / cost["bytes"]
    achieved = cost["flops"] / seconds
    return {
        "name": name,
        "rows": rows,
        "flops": cost["flops"],
        "bytes": cost["bytes"],
        "intensity": intensity,
        "us": seconds * 1e6,
        "tflops": achieved / 1e12,
        "gb_s": cost["bytes"] / seconds / 1e9,
        "pct_of_roof": 100.0 * achieved / roof(intensity, peak_gb_s, peak_tflops),
    }


def op_points(shape, profiles, context, peak_gb_s, peak_tflops):
    points = []
    for rows, text in profiles:
        times = parse_op_table(text)
        costs = op_costs(shape, rows, context)
        for op in OPS:
            if times.get(op, 0.0) > 0.0:
                points.append(point(op, rows, costs[op], times[op] * 1e-6, peak_gb_s, peak_tflops))
    return points


def step_points(shape, rows, context, peak_gb_s, peak_tflops):
    tpot = {}
    for r in rows:
        if r["system"] in STEP_SYSTEMS and not math.isnan(r["tpot_s_p50"]) and r["tpot_s_p50"] > 0:
            tpot.setdefault((r["system"], int(r["max_batch"])), []).append(r["tpot_s_p50"])
    points = []
    for (system, batch), values in sorted(tpot.items(), key=lambda kv: (STEP_SYSTEMS.index(kv[0][0]), kv[0][1])):
        values.sort()
        points.append(point(system, batch, step_cost(shape, batch, context), values[len(values) // 2],
                            peak_gb_s, peak_tflops))
    return points


def markdown(ops, steps, context, peak_gb_s, peak_tflops):
    ridge = peak_tflops * 1e3 / peak_gb_s
    lines = [f"Ceilings: {peak_gb_s:g} GB/s and {peak_tflops:g} TFLOP/s (BF16), ridge point "
             f"{ridge:.0f} FLOP/byte; mean attended context {context:g} tokens.", ""]
    header = "| {} | rows | FLOP/byte | us/step | TFLOP/s | GB/s | % of roof |"
    if ops:
        lines += [header.format("op"), "|---|---|---|---|---|---|---|"]
        for p in ops:
            lines.append(f"| {p['name']} | {p['rows']} | {p['intensity']:.1f} | {p['us']:.1f} | "
                         f"{p['tflops']:.2f} | {p['gb_s']:.0f} | {p['pct_of_roof']:.0f} |")
        lines.append("")
    if steps:
        lines += [header.format("whole decode step").replace("us/step", "TPOT us"),
                  "|---|---|---|---|---|---|---|"]
        for p in steps:
            lines.append(f"| {p['name']} | {p['rows']} | {p['intensity']:.1f} | {p['us']:.0f} | "
                         f"{p['tflops']:.2f} | {p['gb_s']:.0f} | {p['pct_of_roof']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def roofline_plot(ops, steps, peak_gb_s, peak_tflops, out, theme, note=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = plot._style(plt, theme)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=160)
    ridge = peak_tflops * 1e3 / peak_gb_s
    xs = [0.25, ridge, ridge * 8]
    ax.plot(xs, [roof(x, peak_gb_s, peak_tflops) / 1e12 for x in xs], color=t["muted"], linewidth=1.8)
    ax.text(0.3, roof(0.3, peak_gb_s, peak_tflops) / 1e12 * 0.55, f"{peak_gb_s:g} GB/s", color=t["muted"],
            fontsize=9, rotation=33, rotation_mode="anchor")
    ax.text(ridge * 1.1, peak_tflops * 1.08, f"{peak_tflops:g} TFLOP/s BF16", color=t["muted"], fontsize=9)
    ax.axvline(ridge, color=t["grid"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(ridge * 0.92, 0.012, f"ridge {ridge:.0f} FLOP/byte", color=t["muted"], fontsize=8,
            ha="right")

    batches = sorted({p["rows"] for p in ops})
    markers = ["o", "s", "D", "^"]
    for i, rows in enumerate(batches):
        group = [p for p in ops if p["rows"] == rows]
        ax.scatter([p["intensity"] for p in group], [p["tflops"] for p in group], s=46,
                   marker=markers[i % len(markers)], facecolor="none", edgecolor=t["text"], linewidth=1.4,
                   label=f"ops, {rows} row{'s' if rows > 1 else ''}", zorder=3)
        logs = plot.spread_labels([math.log10(p["tflops"]) for p in group], 0.1)
        for p, y in zip(group, logs):
            ax.annotate(p["name"], (p["intensity"], p["tflops"]), xytext=(p["intensity"] / 1.6, 10 ** y),
                        textcoords="data", ha="right", va="center", color=t["muted"], fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": t["grid"], "linewidth": 0.8})

    plotted = [s for s in PLOT_SYSTEMS if any(p["name"] == s for p in steps)]
    for system in plotted:
        group = [p for p in steps if p["name"] == system]
        ax.plot([p["intensity"] for p in group], [p["tflops"] for p in group], color=t["series"][system],
                linewidth=1.8, marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5,
                label=f"{system}, whole step", zorder=4)
        if system == plotted[-1]:
            for p in group:
                ax.annotate(f"batch {p['rows']}", (p["intensity"], p["tflops"]), xytext=(0, -15),
                            textcoords="offset points", ha="center", color=t["muted"], fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, ridge * 8)
    ax.set_ylim(0.01, peak_tflops * 2)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("Decode roofline")
    ax.legend(frameon=False, loc="upper left", labelcolor=t["text"], fontsize=9)
    if note:
        fig.text(0.01, 0.01, note, color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def parse_profile_arg(value):
    path, sep, rows = value.rpartition(":")
    if not sep or not rows.isdigit():
        raise argparse.ArgumentTypeError(f"expected <file>:<rows>, got '{value}'")
    return Path(path), int(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Place the decode step and its ops on a roofline")
    ap.add_argument("summaries", nargs="*", type=Path, help="*_summary.csv files for the whole-step points")
    ap.add_argument("--profile", action="append", default=[], type=parse_profile_arg,
                    help="<profile_report.py output>:<rows per decode step>, repeatable")
    ap.add_argument("--config", type=Path, default=Path("models/tinyllama/config.json"))
    ap.add_argument("--context", type=float, default=96.0,
                    help="mean attended context of a decode step (default 64 prompt + 64 / 2 output)")
    ap.add_argument("--peak-gb-s", type=float, default=PEAK_GB_S)
    ap.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS)
    ap.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--theme", choices=sorted(plot.THEMES), default="light")
    args = ap.parse_args(argv)

    shape = model_shape(json.loads(args.config.read_text()))
    profiles = [(rows, path.read_text()) for path, rows in args.profile if path.exists()]
    ops = op_points(shape, profiles, args.context, args.peak_gb_s, args.peak_tflops)
    rows = plot.load_summaries(args.summaries)
    steps = step_points(shape, rows, args.context, args.peak_gb_s, args.peak_tflops)
    if not ops and not steps:
        ap.error("no op timings and no summaries to place")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpus = sorted({r.get("gpu", "") for r in rows if r.get("gpu")})
    png = args.out_dir / "roofline.png"
    roofline_plot(ops, steps, args.peak_gb_s, args.peak_tflops, png, args.theme,
                  f"{', '.join(gpus)}; whole step from median TPOT" if gpus else None)
    md = args.out_dir / "roofline.md"
    md.write_text(markdown(ops, steps, args.context, args.peak_gb_s, args.peak_tflops))
    print(f"wrote {png}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
