#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_csv as bc  # noqa: E402

WRAPPED_NS = 2**63


def run_llama_bench(binary, model, prompt_len, output_len, repetitions, extra):
    cmd = [binary, "-m", str(model), "-p", str(prompt_len), "-n", str(output_len),
           "-r", str(repetitions), "-o", "json", *extra]
    done = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_llama_bench(done.stdout)


def parse_llama_bench(text):
    rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError("llama-bench printed no results")
    return rows


def samples(row):
    tokens = row.get("n_prompt", 0) + row.get("n_gen", 0)
    durations = row.get("samples_ns") or []
    if durations:
        kept = [int(ns) for ns in durations if 0 < int(ns) < WRAPPED_NS]
        if len(kept) < len(durations):
            print(f"dropped {len(durations) - len(kept)} llama-bench sample(s) with a negative duration",
                  file=sys.stderr)
        if not kept:
            raise ValueError("every llama-bench sample has a negative duration")
        return [tokens * 1e9 / ns for ns in kept]
    values = row.get("samples_ts") or []
    if values:
        return [float(v) for v in values]
    if "avg_ts" in row:
        return [float(row["avg_ts"])]
    raise ValueError("llama-bench row has neither samples_ts nor avg_ts")


def summary_from_rows(rows, prompt_len, output_len):
    prefill = [r for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0]
    decode = [r for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0]
    if not prefill or not decode:
        raise ValueError("llama-bench output needs one prompt test (-p) and one generation test (-n)")
    pp = samples(prefill[0])
    tg = samples(decode[0])
    ttft = [prompt_len / s for s in pp]
    tpot = [1.0 / s for s in tg]
    count = max(len(ttft), len(tpot))
    latency = [ttft[i % len(ttft)] + (output_len - 1) * tpot[i % len(tpot)] for i in range(count)]

    def stat(values):
        return {"mean": sum(values) / len(values), "p50": bc.percentile(values, 50),
                "p90": bc.percentile(values, 90), "p99": bc.percentile(values, 99)}

    wall = stat(latency)["mean"]
    fields = {
        "requests": count,
        "prompt_tokens": prompt_len * count,
        "output_tokens": output_len * count,
        "wall_s": wall * count,
        "request_throughput": 1.0 / wall,
        "output_throughput": output_len / wall,
    }
    for name, values in (("ttft_s", ttft), ("tpot_s", tpot), ("latency_s", latency)):
        for key, value in stat(values).items():
            fields[f"{name}_{key}"] = value
    fields.update(bc.bandwidth_fields(fields))
    return fields


def main(argv=None):
    ap = argparse.ArgumentParser(description="llama.cpp batch-1 baseline through llama-bench")
    ap.add_argument("--model", type=Path, help="GGUF file (BF16 to match the engine)")
    ap.add_argument("--llama-bench", default="llama-bench", help="path to the llama-bench binary")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--from-json", type=Path, help="parse a saved `llama-bench -o json` output instead of running")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("extra", nargs="*", help="more llama-bench options after --, e.g. -ngl 99")
    args = ap.parse_args(argv)

    if args.from_json:
        rows = parse_llama_bench(args.from_json.read_text())
    elif args.model:
        rows = run_llama_bench(args.llama_bench, args.model, args.prompt_len, args.output_len,
                               args.repetitions, args.extra or ["-ngl", "99"])
    else:
        ap.error("give --model or --from-json")

    first = rows[0]
    summary = summary_from_rows(rows, args.prompt_len, args.output_len)
    settings = {
        "version": f"llama.cpp {first.get('build_commit', '?')} (build {first.get('build_number', '?')})",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": first.get("gpu_info", "?"),
        "build": ",".join(str(b) for b in [first.get("backends", "?")]),
        "model": Path(first.get("model_filename", str(args.model))).name,
        "driver": "llamacpp_bench",
        "max_batch": 1,
        "num_requests": summary["requests"],
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "request_rate": 0,
        "warmup": 0,
        "seed": 0,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "stop_at_eos": 0,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    bc.write_summary_csv(args.summary, settings, summary)
    print(f"prefill {args.prompt_len / summary['ttft_s_p50']:.0f} tokens/s, "
          f"decode {1.0 / summary['tpot_s_p50']:.1f} tokens/s "
          f"({summary['bandwidth_gb_s']:.0f} GB/s of weights)")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
