#!/usr/bin/env python3
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_csv as bc  # noqa: E402
import bench_workload as bw  # noqa: E402


def greedy_batch(model, torch, prompts, steps, device):
    batch = len(prompts)
    width = max(len(p) for p in prompts)
    ids = torch.zeros((batch, width), dtype=torch.long)
    mask = torch.zeros((batch, width), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):] = torch.tensor(p, dtype=torch.long)
        mask[i, width - len(p):] = 1
    ids, mask = ids.to(device), mask.to(device)
    position = (mask.cumsum(-1) - 1).clamp(min=0)

    tokens = [[] for _ in range(batch)]
    times = [[] for _ in range(batch)]
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        last_position = position[:, -1]
        for step in range(steps):
            host = nxt.cpu().tolist()
            now = time.perf_counter()
            for i in range(batch):
                tokens[i].append(host[i])
                times[i].append(now)
            if step == steps - 1:
                break
            mask = torch.cat([mask, torch.ones((batch, 1), dtype=mask.dtype, device=device)], dim=1)
            last_position = last_position + 1
            out = model(input_ids=nxt[:, None], attention_mask=mask,
                        position_ids=last_position[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
    return tokens, times


def run(model, torch, requests, max_batch, device, start):
    metrics = []
    for first in range(0, len(requests), max_batch):
        group = requests[first:first + max_batch]
        ready = start + max(r["arrival"] for r in group)
        delay = ready - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        steps = max(r["max_new_tokens"] for r in group)
        _, times = greedy_batch(model, torch, [r["prompt"] for r in group], steps, device)
        for r, t in zip(group, times):
            n = r["max_new_tokens"]
            metrics.append(bc.request_metrics(r["id"], len(r["prompt"]), n, r["arrival"],
                                              t[0] - start, t[n - 1] - start))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="HuggingFace transformers baseline for engine bench")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--prompt-len", default="128")
    ap.add_argument("--output-len", default="256")
    ap.add_argument("--request-rate", type=float, default=0.0)
    ap.add_argument("--max-batch", type=int, default=1,
                    help="1 serves one request at a time; N serves static batches of N")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa", help="attn_implementation: sdpa, eager, flash_attention_2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    prompt_len = bw.parse_length_range(args.prompt_len)
    output_len = bw.parse_length_range(args.output_len)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn)
    model = model.to(args.device).eval()
    cfg = model.config
    kw = dict(vocab=cfg.vocab_size, bos_id=cfg.bos_token_id, eos_id=cfg.eos_token_id)

    if args.warmup > 0:
        warm = bw.make_workload(args.warmup, prompt_len, output_len, seed=args.seed + 1, **kw)
        run(model, torch, warm, args.max_batch, args.device, time.perf_counter())
        torch.cuda.synchronize()

    requests = bw.make_workload(args.num_requests, prompt_len, output_len,
                                request_rate=args.request_rate, seed=args.seed, **kw)
    start = time.perf_counter()
    metrics = run(model, torch, requests, args.max_batch, args.device, start)
    torch.cuda.synchronize()
    end = max(m["arrival_s"] + m["latency_s"] for m in metrics)
    summary = bc.summarize(metrics, end)

    settings = {
        "version": f"transformers {transformers.__version__}, torch {torch.__version__}",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else platform.processor(),
        "build": f"attn={args.attn}",
        "model": args.model.name,
        "driver": "hf_sequential" if args.max_batch == 1 else "hf_static_batch",
        "max_batch": args.max_batch,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "request_rate": args.request_rate,
        "warmup": args.warmup,
        "seed": args.seed,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "stop_at_eos": 0,
    }
    if args.max_batch == 1:
        summary.update(bc.bandwidth_fields(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bc.write_request_csv(args.out, metrics)
    summary_path = args.summary or args.out.with_name(args.out.stem + "_summary" + args.out.suffix)
    bc.write_summary_csv(summary_path, settings, summary)
    print(f"{len(metrics)} requests, {summary['output_tokens']} output tokens in "
          f"{summary['wall_s']:.2f} s: {summary['output_throughput']:.1f} tokens/s")
    print(f"TTFT p50 {summary['ttft_s_p50'] * 1e3:.1f} ms, TPOT p50 {summary['tpot_s_p50'] * 1e3:.2f} ms")
    print(f"wrote {args.out} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
