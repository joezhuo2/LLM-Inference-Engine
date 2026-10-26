import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import profile_report

KERNELS_PER_LAYER = 10


def quantile(values, q):
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def median_step(path, layer):
    conn = sqlite3.connect(path)
    strings = profile_report.string_table(conn)
    kernels = profile_report.attribute(profile_report.load_kernels(conn, strings), profile_report.load_ranges(conn, strings))
    api_end = dict(
        conn.execute(
            "SELECT k.start, r.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId"
        )
    )
    steps = defaultdict(list)
    for k in kernels:
        if k["kind"] == "forward_decode":
            steps[k["step"]].append(k)
    for ks in steps.values():
        ks.sort(key=lambda k: k["start"])

    gaps = [(b["start"] - a["end"]) / 1e3 for ks in steps.values() for a, b in zip(ks, ks[1:])]
    ordered = sorted(steps.values(), key=lambda ks: ks[-1]["end"] - ks[0]["start"])
    ks = ordered[len(ordered) // 2]

    first = 1 + KERNELS_PER_LAYER * layer
    window = ks[first : first + KERNELS_PER_LAYER]
    if len(window) < KERNELS_PER_LAYER:
        raise SystemExit(f"{path}: the median step has no layer {layer}")
    t0 = window[0]["start"]
    t1 = window[-1]["end"]

    launches = conn.execute(
        """
        SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId
        WHERE r.end >= ? AND r.start <= ? AND (s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cudaGraphLaunch%')
        """,
        (t0, t1),
    ).fetchall()
    calls = defaultdict(list)
    for start, end, name in conn.execute(
        "SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId WHERE s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cudaGraphLaunch%'"
    ):
        calls[name.split("_v")[0]].append((end - start) / 1e3)
    ahead = [(k["start"] - api_end[k["start"]]) / 1e3 for k in ks if k["start"] in api_end]
    span = (ks[-1]["end"] - ks[0]["start"]) / 1e3
    busy, _, _ = profile_report.busy_and_span([(k["start"], k["end"]) for k in ks])

    return {
        "steps": len(steps),
        "kernels": len(ks),
        "span_us": round(span, 1),
        "busy_us": round(busy / 1e3, 1),
        "gap_us": {q: round(quantile(gaps, v), 2) for q, v in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))},
        "launches": {name: {"per_step": round(len(v) / len(steps), 1), "median_us": round(statistics.median(v), 1)} for name, v in sorted(calls.items())},
        "cpu_ahead_us": round(statistics.median(ahead), 1),
        "layer": layer,
        "kernels_in_layer": [{"op": k["op"], "start": round((k["start"] - t0) / 1e3, 2), "end": round((k["end"] - t0) / 1e3, 2)} for k in window],
        "cpu_in_layer": [
            {"name": name.split("_v")[0], "start": round((s - t0) / 1e3, 2), "end": round((e - t0) / 1e3, 2)} for s, e, name in sorted(launches)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="One layer of the median decode step, eager and replayed from a CUDA graph, from two Nsight Systems exports")
    parser.add_argument("eager", help="sqlite export of the eager profile (scripts/profile.sh nsys decode_step)")
    parser.add_argument("graphs", help="sqlite export of the --graphs profile")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--commit", required=True, help="commit the profiles were taken on")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = {
        "commit": args.commit,
        "files": {"eager": Path(args.eager).name, "graphs": Path(args.graphs).name},
        "eager": median_step(args.eager, args.layer),
        "graphs": median_step(args.graphs, args.layer),
    }
    ops = [k["op"] for k in data["eager"]["kernels_in_layer"]]
    if ops[0] != "add2_norm1" or ops[-1] != "down":
        raise SystemExit(f"layer {args.layer} of the eager step runs {ops}, not add2_norm1 to down")
    for k, op in zip(data["graphs"]["kernels_in_layer"], ops):
        k["op"] = op
    args.out.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
