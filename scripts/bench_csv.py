import csv
import math

REQUEST_COLUMNS = [
    "id", "prompt_tokens", "output_tokens", "preemptions", "arrival_s", "ttft_s", "tpot_s",
    "latency_s",
]
DECODE_WEIGHT_BYTES = 2_069_024_768


def percentile(values, q):
    values = sorted(v for v in values if not math.isnan(v))
    if not values:
        return math.nan
    rank = q / 100.0 * (len(values) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (rank - lo) * (values[hi] - values[lo])


def request_metrics(request_id, prompt_tokens, output_tokens, arrival, first_token, finished):
    return {
        "id": request_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "preemptions": 0,
        "arrival_s": arrival,
        "ttft_s": first_token - arrival,
        "tpot_s": (finished - first_token) / (output_tokens - 1) if output_tokens > 1 else math.nan,
        "latency_s": finished - arrival,
    }


def _stat(values):
    values = [v for v in values if not math.isnan(v)]
    return {
        "mean": sum(values) / len(values) if values else math.nan,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def summarize(metrics, wall_s):
    if not metrics:
        raise ValueError("summarize: no requests")
    if not wall_s > 0.0 or math.isinf(wall_s):
        raise ValueError("summarize: wall_s must be positive and finite")
    out_tokens = sum(m["output_tokens"] for m in metrics)
    fields = {
        "requests": len(metrics),
        "prompt_tokens": sum(m["prompt_tokens"] for m in metrics),
        "output_tokens": out_tokens,
        "wall_s": wall_s,
        "request_throughput": len(metrics) / wall_s,
        "output_throughput": out_tokens / wall_s,
    }
    for name in ("ttft_s", "tpot_s", "latency_s"):
        for stat, value in _stat([m[name] for m in metrics]).items():
            fields[f"{name}_{stat}"] = value
    return fields


def bandwidth_fields(summary, weight_bytes=DECODE_WEIGHT_BYTES):
    tpot = summary["tpot_s_p50"]
    return {
        "decode_weight_bytes": weight_bytes,
        "bandwidth_gb_s": weight_bytes / tpot / 1e9 if tpot and not math.isnan(tpot) else math.nan,
    }


def _number(value):
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.9g}"
    return str(value)


def write_request_csv(path, metrics):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(REQUEST_COLUMNS)
        for m in metrics:
            w.writerow([_number(m[c]) for c in REQUEST_COLUMNS])


def write_summary_csv(path, settings, summary):
    row = {**settings, **summary}
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(row.keys())
        w.writerow([_number(v) for v in row.values()])
