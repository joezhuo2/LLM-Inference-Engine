#!/usr/bin/env python3
import argparse
import bisect
import csv
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OPS = (
    "embedding", "norm1", "qkv", "rope", "write_kv", "attention", "o_proj", "add1",
    "norm2", "gate_up", "swiglu", "down", "add2", "final_norm", "lm_head",
)
STEP_KINDS = ("forward_decode", "forward_prefill")
LAUNCH_GAP_US = 5.0
DECODE_WEIGHT_BYTES = 2_069_024_768


def export_sqlite(report: Path) -> Path:
    out = report.with_suffix(".sqlite")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(out), str(report)],
        check=True, capture_output=True,
    )
    return out


def string_table(conn):
    return dict(conn.execute("SELECT id, value FROM StringIds"))


def load_ranges(conn, strings):
    rows = conn.execute(
        "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ).fetchall()
    ranges = []
    for start, end, tid, text, text_id in rows:
        name = text if text is not None else strings.get(text_id)
        if name is not None:
            ranges.append((start, end, tid, name))
    return ranges


def load_kernels(conn, strings):
    rows = conn.execute(
        """
        SELECT k.start, k.end, k.shortName, k.demangledName, r.start, r.globalTid
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        """
    ).fetchall()
    kernels = []
    for start, end, short_id, demangled_id, launch, tid in rows:
        name = strings.get(demangled_id) or strings.get(short_id) or "?"
        kernels.append({"start": start, "end": end, "name": name, "launch": launch, "tid": tid})
    return kernels


def attribute(kernels, ranges):
    by_tid = defaultdict(list)
    for r in ranges:
        by_tid[r[2]].append(r)
    for tid_ranges in by_tid.values():
        tid_ranges.sort(key=lambda r: (r[0], -r[1]))
    starts = {tid: [r[0] for r in rs] for tid, rs in by_tid.items()}

    for k in kernels:
        k["op"] = None
        k["step"] = None
        k["kind"] = None
        rs = by_tid.get(k["tid"], [])
        hi = bisect.bisect_right(starts.get(k["tid"], []), k["launch"])
        enclosing = [r for r in rs[:hi] if r[1] >= k["launch"]]
        for r in enclosing:
            if r[3] in OPS and (k["op"] is None):
                k["op"] = r[3]
            if r[3] in STEP_KINDS:
                k["step"] = (r[0], r[1])
                k["kind"] = r[3]
        if k["op"] is None and enclosing:
            inner = min(enclosing, key=lambda r: r[1] - r[0])
            k["op"] = inner[3]
    return kernels


def busy_and_span(intervals):
    intervals = sorted(intervals)
    busy = 0
    cursor = None
    gaps = []
    first = intervals[0][0]
    last = max(e for _, e in intervals)
    for s, e in intervals:
        if cursor is None or s > cursor:
            if cursor is not None:
                gaps.append(s - cursor)
            busy += e - s
            cursor = e
        elif e > cursor:
            busy += e - cursor
            cursor = e
    return busy, last - first, gaps


def step_table(kernels):
    steps = defaultdict(list)
    for k in kernels:
        if k["step"] is not None:
            steps[(k["kind"], k["step"])].append(k)
    rows = []
    for (kind, window), ks in sorted(steps.items(), key=lambda kv: kv[0][1]):
        busy, span, gaps = busy_and_span([(k["start"], k["end"]) for k in ks])
        rows.append({
            "kind": kind,
            "kernels": len(ks),
            "span_us": span / 1e3,
            "busy_us": busy / 1e3,
            "idle_us": (span - busy) / 1e3,
            "idle_pct": 100.0 * (span - busy) / span if span else 0.0,
            "gaps_over_5us": sum(1 for g in gaps if g / 1e3 > LAUNCH_GAP_US),
        })
    return rows


def op_table(kernels):
    total = defaultdict(int)
    count = defaultdict(int)
    for k in kernels:
        if k["step"] is None or k["kind"] != "forward_decode":
            continue
        total[k["op"] or "(none)"] += k["end"] - k["start"]
        count[k["op"] or "(none)"] += 1
    decode_steps = len({k["step"] for k in kernels if k["kind"] == "forward_decode"}) or 1
    grand = sum(total.values()) or 1
    rows = []
    for op, ns in sorted(total.items(), key=lambda kv: -kv[1]):
        rows.append({
            "op": op,
            "kernels_per_step": count[op] / decode_steps,
            "us_per_step": ns / 1e3 / decode_steps,
            "pct": 100.0 * ns / grand,
        })
    return rows


def kernel_table(kernels, top):
    total = defaultdict(int)
    count = defaultdict(int)
    for k in kernels:
        total[k["name"]] += k["end"] - k["start"]
        count[k["name"]] += 1
    grand = sum(total.values()) or 1
    rows = []
    for name, ns in sorted(total.items(), key=lambda kv: -kv[1])[:top]:
        rows.append({
            "kernel": name if len(name) <= 80 else name[:77] + "...",
            "count": count[name],
            "total_ms": ns / 1e6,
            "avg_us": ns / 1e3 / count[name],
            "pct": 100.0 * ns / grand,
        })
    return rows


def bandwidth_section(summary_path: Path, peak_gb_s: float, weight_bytes: float, decode_busy_us):
    with open(summary_path, newline="") as f:
        row = next(csv.DictReader(f))
    tpot = float(row["tpot_s_p50"])
    weight_bytes = float(row.get("decode_weight_bytes") or weight_bytes)
    achieved = weight_bytes / tpot / 1e9
    out = {
        "tpot_ms": tpot * 1e3,
        "achieved_gb_s": achieved,
        "pct_of_peak": 100.0 * achieved / peak_gb_s,
        "target_pct": 60.0,
        "max_batch": row.get("max_batch", "?"),
    }
    if decode_busy_us:
        busy_s = decode_busy_us * 1e-6
        out["gpu_busy_ms"] = busy_s * 1e3
        out["busy_pct_of_tpot"] = 100.0 * busy_s / tpot
        if weight_bytes:
            out["kernel_gb_s"] = weight_bytes / busy_s / 1e9
    return out


def table(rows, columns):
    header = "| " + " | ".join(c for c, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[key]) for _, key in columns) + " |")
    return "\n".join(lines)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def report(conn, top=12, summary=None, peak=323.6, weight_bytes=DECODE_WEIGHT_BYTES):
    strings = string_table(conn)
    kernels = attribute(load_kernels(conn, strings), load_ranges(conn, strings))
    out = []
    steps = step_table(kernels)
    decode = [s for s in steps if s["kind"] == "forward_decode"]
    prefill = [s for s in steps if s["kind"] == "forward_prefill"]

    out.append("## Launch gaps per forward step")
    out.append("")
    for name, group in (("decode", decode), ("prefill", prefill)):
        if not group:
            continue
        out.append(
            f"- {name}: {len(group)} steps, {statistics.median(s['kernels'] for s in group):.0f} kernels, "
            f"median span {statistics.median(s['span_us'] for s in group):.0f} us, "
            f"median busy {statistics.median(s['busy_us'] for s in group):.0f} us, "
            f"median idle {statistics.median(s['idle_pct'] for s in group):.1f}% of the span, "
            f"{statistics.median(s['gaps_over_5us'] for s in group):.0f} gaps over {LAUNCH_GAP_US:g} us per step"
        )
    if not steps:
        out.append("- no forward_decode or forward_prefill ranges with kernels found; "
                   "profile with `engine bench --max-batch N --profile`")
    out.append("")

    ops = op_table(kernels)
    if ops:
        out.append("## GPU time per op in a decode step")
        out.append("")
        out.append(table(ops, [("op", "op"), ("kernels/step", "kernels_per_step"),
                               ("GPU us/step", "us_per_step"), ("% of kernel time", "pct")]))
        out.append("")

    out.append(f"## Top {top} kernels")
    out.append("")
    out.append(table(kernel_table(kernels, top), [("kernel", "kernel"), ("count", "count"),
                                                  ("total ms", "total_ms"), ("avg us", "avg_us"),
                                                  ("%", "pct")]))
    out.append("")

    if summary:
        busy = statistics.median(s["busy_us"] for s in decode) if decode else None
        b = bandwidth_section(summary, peak, weight_bytes, busy)
        out.append("## Decode bandwidth")
        out.append("")
        out.append(f"- max_batch {b['max_batch']}, median TPOT {b['tpot_ms']:.2f} ms")
        out.append(f"- weights read per token / TPOT: {b['achieved_gb_s']:.1f} GB/s = "
                   f"{b['pct_of_peak']:.1f}% of the {peak:g} GB/s peak (target {b['target_pct']:.0f}%)")
        if "gpu_busy_ms" in b:
            out.append(f"- GPU kernel busy time in a decode step: {b['gpu_busy_ms']:.2f} ms "
                       f"({b['busy_pct_of_tpot']:.0f}% of TPOT)")
        if "kernel_gb_s" in b:
            out.append(f"- weights / kernel busy time: {b['kernel_gb_s']:.1f} GB/s "
                       "(what the kernels would reach with no launch gaps)")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize an nsys capture of `engine bench --profile`")
    ap.add_argument("capture", type=Path, help=".nsys-rep or .sqlite file")
    ap.add_argument("--summary", type=Path, help="the bench *_summary.csv of the same run")
    ap.add_argument("--peak-gb-s", type=float, default=323.6, help="measured DRAM bandwidth")
    ap.add_argument("--weight-bytes", type=float, default=DECODE_WEIGHT_BYTES,
                    help="bytes of weights one decode step reads (default: TinyLlama, 2.07 GB)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    path = args.capture
    if path.suffix == ".nsys-rep":
        path = export_sqlite(path)
    conn = sqlite3.connect(path)
    print(report(conn, args.top, args.summary, args.peak_gb_s, args.weight_bytes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
