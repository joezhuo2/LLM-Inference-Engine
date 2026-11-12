#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def lookup(context, max_ngram, max_draft):
    size = len(context)
    for g in range(min(max_ngram, size - 1), 0, -1):
        pattern = context[size - g:]
        for i in range(size - 1 - g, -1, -1):
            if context[i:i + g] == pattern:
                return g, i, context[i + g:i + g + max_draft]
    return 0, -1, []


def main():
    parser = argparse.ArgumentParser(description="Rebuild every speculative step of a one-request engine bench trace (the match, the draft, what was kept) for the site, and check it against the counts the engine recorded")
    parser.add_argument("--trace", required=True, help="trace JSON of engine bench --speculate N --num-requests 1, for example results/speculative/traces/repeat_fix_typos_k8.json")
    parser.add_argument("--prompts", default=str(ROOT / "scripts/prompts_repeat.json"))
    parser.add_argument("--name", required=True, help="the name of the run's prompt in --prompts")
    parser.add_argument("--model", default=str(ROOT / "models/tinyllama"))
    parser.add_argument("--max-ngram", type=int, default=3, help="SchedulerConfig::max_ngram of the run")
    parser.add_argument("--out", default=str(ROOT / "results/speculative_example.json"))
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    meta = trace["meta"]
    if len(trace["requests"]) != 1:
        raise SystemExit("the trace must hold one request: the draft counts of a step are not split by sequence")
    request = trace["requests"][0]
    named = [p for p in json.loads(Path(args.prompts).read_text())["prompts"] if p["name"] == args.name]
    if len(named) != 1:
        raise SystemExit(f"{args.prompts} has no prompt named {args.name}")
    prompt = named[0]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok.apply_chat_template(prompt["messages"], add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = [tok.bos_token_id] + list(prompt_ids)
    if len(prompt_ids) != request["prompt_tokens"]:
        raise SystemExit(f"the engine counted {request['prompt_tokens']} prompt tokens, BOS and the HF template give {len(prompt_ids)}")

    if ":" in meta["output_len"]:
        raise SystemExit("the run must have one --output-len, not a range")
    max_new_tokens = int(meta["output_len"])
    context = list(prompt_ids)
    output = []
    steps = []
    for step in trace["steps"]:
        ids = [t["id"] for t in step["tokens"]]
        if step["kind"] == "decode":
            left = max_new_tokens - len(output) - 1
            cap = min(meta["speculate"], meta["max_batch"] - 1, left)
            ngram, match, draft = lookup(context, args.max_ngram, cap) if cap > 0 else (0, -1, [])
            kept = len(ids) - 1
            if len(draft) != step["drafted"] or kept != step["accepted"]:
                raise SystemExit(f"step {step['index']}: rebuilt a draft of {len(draft)} with {kept} kept, the engine recorded {step['drafted']} and {step['accepted']}")
            if draft[:kept] != ids[:kept]:
                raise SystemExit(f"step {step['index']}: the kept tokens are not the start of the rebuilt draft")
            steps.append({"index": step["index"], "at": len(context), "ngram": ngram, "match": match, "draft": tok.convert_ids_to_tokens(draft), "kept": kept, "ms": round((step["end_s"] - step["start_s"]) * 1e3, 3)})
        context += ids
        output += ids
    if len(output) != request["output_tokens"]:
        raise SystemExit(f"the steps hold {len(output)} output tokens, the request {request['output_tokens']}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {
        "date": meta["date"],
        "gpu": meta["gpu"],
        "commit": commit,
        "name": prompt["name"],
        "max_draft": meta["speculate"],
        "max_ngram": args.max_ngram,
        "max_batch": meta["max_batch"],
        "prompt": prompt["messages"][0]["content"],
        "prompt_tokens": tok.convert_ids_to_tokens(prompt_ids),
        "output_tokens": tok.convert_ids_to_tokens(output),
        "steps": steps,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    drafted = sum(len(s["draft"]) for s in steps)
    kept = sum(s["kept"] for s in steps)
    print(f"{prompt['name']}: {len(prompt_ids)} prompt + {len(output)} output tokens in {len(steps)} decode steps, {kept} of {drafted} draft tokens kept")


if __name__ == "__main__":
    main()
