#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_csv as bc  # noqa: E402
import bench_workload as bw  # noqa: E402

PREEMPTIONS = re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)\s*$", re.M)
DTYPE = re.compile(r"dtype=torch\.(\w+)")
ATTENTION = re.compile(r"Using (\w+) attention backend")


def completion_payload(request, model):
    return {
        "model": model,
        "prompt": request["prompt"],
        "max_tokens": request["max_new_tokens"],
        "min_tokens": request["max_new_tokens"],
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }


def parse_sse(line):
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    data = line[len(b"data:"):].strip()
    if data == b"[DONE]":
        return None
    return json.loads(data)


class TokenClock:
    def __init__(self):
        self.tokens = 0
        self.first = None
        self.last = None
        self.chunks = 0
        self.multi_token_chunks = 0

    def observe(self, event, now):
        usage = event.get("usage") or {}
        count = usage.get("completion_tokens")
        if count is None or count <= self.tokens:
            return
        self.chunks += 1
        if count - self.tokens > 1:
            self.multi_token_chunks += 1
        if self.first is None:
            self.first = now
        self.last = now
        self.tokens = count


def request_metric(request, clock, start):
    if clock.tokens != request["max_new_tokens"]:
        raise RuntimeError(f"request {request['id']}: {clock.tokens} tokens streamed, "
                           f"{request['max_new_tokens']} asked for")
    return bc.request_metrics(request["id"], len(request["prompt"]), clock.tokens, request["arrival"],
                              clock.first - start, clock.last - start)


def client_stats(lags, clocks):
    return {
        "client_lag_s_p99": bc.percentile(lags, 99),
        "client_lag_s_max": max(lags),
        "stream_chunks": sum(c.chunks for c in clocks),
        "multi_token_chunks": sum(c.multi_token_chunks for c in clocks),
    }


def parse_preemptions(text):
    return int(sum(float(v) for v in PREEMPTIONS.findall(text)))


def parse_server_log(text):
    dtype = DTYPE.search(text)
    attention = ATTENTION.search(text)
    return {
        "dtype": dtype.group(1) if dtype else "",
        "attention_backend": attention.group(1) if attention else "",
    }


def server_command(binary, model, port, max_num_seqs, gpu_fraction, max_model_len):
    return [str(binary), "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--dtype", "bfloat16", "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_fraction), "--max-model-len", str(max_model_len),
            "--generation-config", "vllm"]


def parse_rates(text):
    rates = [float(r) for r in text.split(",")]
    if not rates or any(r <= 0.0 for r in rates):
        raise ValueError(f"rates '{text}': expected positive numbers separated by commas")
    return rates


def rate_name(rate):
    return f"{rate:g}".replace(".", "p")


async def send(session, url, payload, request, start, lags, clocks):
    delay = start + request["arrival"] - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    lags.append(time.perf_counter() - start - request["arrival"])
    clock = TokenClock()
    clocks.append(clock)
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for line in response.content:
            event = parse_sse(line)
            if event is not None:
                clock.observe(event, time.perf_counter())
    return request_metric(request, clock, start)


async def drive(base_url, model, requests):
    import aiohttp

    url = f"{base_url}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    lags = []
    clocks = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()
        metrics = await asyncio.gather(*[
            send(session, url, completion_payload(r, model), r, start, lags, clocks) for r in requests
        ])
    return metrics, lags, clocks


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def wait_for_server(base_url, process, timeout_s=1800.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {process.returncode}")
        try:
            fetch(f"{base_url}/health")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1.0)
    raise RuntimeError(f"vllm serve did not answer within {timeout_s:.0f} s")


def stop_server(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def gpu_name():
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.splitlines()[0]


def run_point(args, base_url, model_id, kw, rate, run_index, prompt_len, output_len, log_fields):
    if args.warmup > 0:
        warm = bw.make_workload(args.warmup, prompt_len, output_len, seed=args.seed + 1, **kw)
        asyncio.run(drive(base_url, model_id, warm))
    before = parse_preemptions(fetch(f"{base_url}/metrics"))
    requests = bw.make_workload(args.num_requests, prompt_len, output_len, request_rate=rate,
                                seed=args.seed, **kw)
    metrics, lags, clocks = asyncio.run(drive(base_url, model_id, requests))
    preemptions = parse_preemptions(fetch(f"{base_url}/metrics")) - before
    wall = max(m["arrival_s"] + m["latency_s"] for m in metrics)
    summary = bc.summarize(metrics, wall)
    settings = {
        "version": f"vllm {metadata.version('vllm')}, torch {metadata.version('torch')}",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_name(),
        "build": "Release",
        "model": args.model.name,
        "driver": "vllm_online",
        "max_batch": args.max_num_seqs,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "request_rate": rate,
        "warmup": args.warmup,
        "seed": args.seed,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "stop_at_eos": 0,
    }
    extra = {"preemptions": preemptions, **client_stats(lags, clocks), **log_fields}
    out = args.out_dir / f"vllm_q{rate_name(rate)}_r{run_index}.csv"
    bc.write_request_csv(out, sorted(metrics, key=lambda m: m["id"]))
    bc.write_summary_csv(out.with_name(out.stem + "_summary.csv"), settings, {**summary, **extra})
    print(f"q={rate:g} run {run_index}: {summary['output_throughput']:.1f} tokens/s, "
          f"TTFT p50 {summary['ttft_s_p50'] * 1e3:.0f} ms p99 {summary['ttft_s_p99'] * 1e3:.0f} ms, "
          f"TPOT p50 {summary['tpot_s_p50'] * 1e3:.1f} ms p99 {summary['tpot_s_p99'] * 1e3:.1f} ms, "
          f"lag max {extra['client_lag_s_max'] * 1e3:.1f} ms, preemptions {preemptions}, "
          f"multi-token chunks {extra['multi_token_chunks']} of {extra['stream_chunks']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="vLLM server baseline for engine bench under load")
    ap.add_argument("--model", type=Path, default=Path("models/tinyllama"))
    ap.add_argument("--vllm", type=Path, default=Path.home() / ".venvs/llmie-vllm/bin/vllm")
    ap.add_argument("--server-url", help="attach to a running server instead of starting one")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--rates", default="1,2,3,4,5,6,8")
    ap.add_argument("--run", type=int, default=1, help="repetition number in the file names")
    ap.add_argument("--num-requests", type=int, default=120)
    ap.add_argument("--prompt-len", default="16:512")
    ap.add_argument("--output-len", default="16:256")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/load"))
    ap.add_argument("--server-log", type=Path, help="default: llmie_vllm_server.log in the temp directory")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_len = bw.parse_length_range(args.prompt_len)
    output_len = bw.parse_length_range(args.output_len)
    cfg = json.loads((args.model / "config.json").read_text())
    kw = dict(vocab=cfg["vocab_size"], bos_id=cfg["bos_token_id"], eos_id=cfg["eos_token_id"])

    process = None
    log_path = args.server_log or Path(tempfile.gettempdir()) / "llmie_vllm_server.log"
    base_url = args.server_url or f"http://127.0.0.1:{args.port}"
    log_fields = {"dtype": "", "attention_backend": ""}
    try:
        if not args.server_url:
            with open(log_path, "w") as log:
                process = subprocess.Popen(
                    server_command(args.vllm, args.model, args.port, args.max_num_seqs,
                                   args.gpu_memory_utilization, args.max_model_len),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_for_server(base_url, process)
        if process is not None:
            log_fields = parse_server_log(log_path.read_text())
            if log_fields["dtype"] != "bfloat16":
                raise RuntimeError(f"vllm serve reports dtype '{log_fields['dtype']}', not bfloat16")
        model_id = json.loads(fetch(f"{base_url}/v1/models"))["data"][0]["id"]
        for rate in parse_rates(args.rates):
            run_point(args, base_url, model_id, kw, rate, args.run, prompt_len, output_len, log_fields)
    finally:
        if process is not None:
            stop_server(process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
