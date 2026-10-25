#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_csv as bc  # noqa: E402

VERSION = re.compile(r"build (\d+), commit ([0-9a-f]+)")
WRAPPED_NS = 2**63
FLASH_ATTN = {-1: "auto", 0: "off", 1: "on"}


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


def stat(values):
    return {"mean": sum(values) / len(values), "p50": bc.percentile(values, 50),
            "p90": bc.percentile(values, 90), "p99": bc.percentile(values, 99)}


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


def batched_command(binary, model, prompt_len, output_len, batches, extra):
    context = max(batches) * (prompt_len + output_len)
    return [binary, "-m", str(model), "-npp", str(prompt_len), "-ntg", str(output_len - 1),
            "-npl", ",".join(str(b) for b in [batches[0], *batches]), "-c", str(context),
            "-b", str(context), "--output-format", "jsonl", *extra]


def parse_batched_bench(text):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    if not rows:
        raise ValueError("llama-batched-bench printed no jsonl results")
    for row in rows:
        missing = [k for k in ("pp", "tg", "pl", "t_pp", "t") if k not in row]
        if missing:
            raise ValueError(f"llama-batched-bench row lacks {', '.join(missing)}")
        if row.get("is_pp_shared"):
            raise ValueError("llama-batched-bench ran with a shared prompt (-pps); each request needs its own")
    return rows


def summary_from_batched_row(row):
    output_tokens = row["tg"] + 1
    metrics = [bc.request_metrics(i, row["pp"], output_tokens, 0.0, row["t_pp"], row["t"])
               for i in range(row["pl"])]
    summary = bc.summarize(metrics, row["t"])
    if row["pl"] == 1:
        summary.update(bc.bandwidth_fields(summary))
    return summary


def tool_version(binary):
    done = subprocess.run([binary, "--version"], capture_output=True, text=True)
    m = VERSION.search(done.stdout + done.stderr)
    return f"llama.cpp {m[2]} (build {m[1]})" if m else "llama.cpp ?"


def gpu_name():
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True)
        return done.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "?"


def settings_for(version, gpu, build, model, driver, batch, requests, prompt_len, output_len, warmup):
    return {
        "version": version,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
        "build": build,
        "model": model,
        "driver": driver,
        "max_batch": batch,
        "num_requests": requests,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "request_rate": 0,
        "warmup": warmup,
        "seed": 0,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "stop_at_eos": 0,
    }


def bench_main(args):
    if args.from_json:
        rows = parse_llama_bench(args.from_json.read_text())
    elif args.model:
        rows = run_llama_bench(args.llama_bench, args.model, args.prompt_len, args.output_len,
                               args.repetitions, args.extra or ["-ngl", "99"])
    else:
        raise SystemExit("bench: give --model or --from-json")

    first = rows[0]
    summary = summary_from_rows(rows, args.prompt_len, args.output_len)
    settings = settings_for(
        f"llama.cpp {first.get('build_commit', '?')} (build {first.get('build_number', '?')})",
        first.get("gpu_info", "?"), f"{first.get('backends', '?')} fa={FLASH_ATTN.get(first.get('flash_attn'), '?')}",
        Path(first.get("model_filename", str(args.model))).name, "llamacpp_bench", 1,
        summary["requests"], args.prompt_len, args.output_len, 0)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    bc.write_summary_csv(args.summary, settings, summary)
    print(f"prefill {args.prompt_len / summary['ttft_s_p50']:.0f} tokens/s, "
          f"decode {1.0 / summary['tpot_s_p50']:.1f} tokens/s "
          f"({summary['bandwidth_gb_s']:.0f} GB/s of weights)")
    print(f"wrote {args.summary}")
    return 0


def batched_main(args):
    batches = [int(b) for b in args.max_batch.split(",")]
    if args.from_jsonl:
        runs = [parse_batched_bench(p.read_text()) for p in args.from_jsonl]
        version = "llama.cpp ?"
    elif args.model:
        cmd = batched_command(args.llama_batched_bench, args.model, args.prompt_len, args.output_len,
                              batches, args.extra or ["-ngl", "99"])
        runs = []
        for _ in range(args.repetitions):
            done = subprocess.run(cmd, check=True, capture_output=True, text=True)
            runs.append(parse_batched_bench(done.stdout))
        version = tool_version(args.llama_batched_bench)
    else:
        raise SystemExit("batched: give --model or --from-jsonl")

    gpu = gpu_name()
    model = args.model.name if args.model else "?"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for r, rows in enumerate(runs, start=1):
        for row in rows[1:]:
            summary = summary_from_batched_row(row)
            settings = settings_for(version, gpu, f"CUDA fa={FLASH_ATTN.get(row.get('flash_attn'), '?')}", model,
                                    "llamacpp_batched", row["pl"], row["pl"], row["pp"],
                                    row["tg"] + 1, 1)
            path = args.out_dir / f"{args.prefix}_b{row['pl']}_r{r}_summary.csv"
            bc.write_summary_csv(path, settings, summary)
            print(f"batch {row['pl']} run {r}: {summary['output_throughput']:.1f} tokens/s, "
                  f"TPOT p50 {summary['tpot_s_p50'] * 1e3:.2f} ms, wrote {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="llama.cpp baselines through llama-bench and llama-batched-bench")
    sub = ap.add_subparsers(dest="tool", required=True)

    b = sub.add_parser("bench", help="batch-1 prefill and decode throughput from llama-bench")
    b.add_argument("--model", type=Path, help="GGUF file (BF16 to match the engine)")
    b.add_argument("--llama-bench", default="llama-bench", help="path to the llama-bench binary")
    b.add_argument("--prompt-len", type=int, default=128)
    b.add_argument("--output-len", type=int, default=256)
    b.add_argument("--repetitions", type=int, default=5)
    b.add_argument("--from-json", type=Path, help="parse a saved `llama-bench -o json` output instead of running")
    b.add_argument("--summary", type=Path, required=True)
    b.add_argument("extra", nargs="*", help="llama-bench options after --, replacing the default -ngl 99")
    b.set_defaults(run=bench_main)

    m = sub.add_parser("batched", help="one static batch per size from llama-batched-bench")
    m.add_argument("--model", type=Path, help="GGUF file (BF16 to match the engine)")
    m.add_argument("--llama-batched-bench", default="llama-batched-bench",
                   help="path to the llama-batched-bench binary")
    m.add_argument("--prompt-len", type=int, default=128)
    m.add_argument("--output-len", type=int, default=256)
    m.add_argument("--max-batch", default="1,4,16", help="comma-separated batch sizes")
    m.add_argument("--repetitions", type=int, default=3)
    m.add_argument("--from-jsonl", type=Path, nargs="+",
                   help="parse saved `--output-format jsonl` outputs (one file per repetition, first row the warm-up) instead of running")
    m.add_argument("--out-dir", type=Path, required=True)
    m.add_argument("--prefix", default="llamacpp")
    m.add_argument("extra", nargs="*", help="llama-batched-bench options after --, replacing the default -ngl 99")
    m.set_defaults(run=batched_main)

    args = ap.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
