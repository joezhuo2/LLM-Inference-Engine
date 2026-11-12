#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"(repeat|chat)_(.+)_k(\d+)\.json")


def tokens(path):
    return [t["id"] for step in json.loads(path.read_text())["steps"] for t in step["tokens"]]


def main():
    parser = argparse.ArgumentParser(description="For every trace of scripts/speculative_sweep.sh whose tokens differ from the run without drafts: the first token that differs, and how far apart HuggingFace's logits of the two tokens are there, in BF16 and FP32")
    parser.add_argument("traces", nargs="?", default=str(ROOT / "results/speculative/traces"))
    parser.add_argument("--model", default=str(ROOT / "models/tinyllama"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = {}
    for name, file in (("repeat", "prompts_repeat.json"), ("chat", "prompts.json")):
        for p in json.loads((ROOT / "scripts" / file).read_text())["prompts"]:
            prompts[(name, p["name"])] = p["messages"]
    models = {dtype: AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda() for dtype in (torch.bfloat16, torch.float32)}

    seen = set()
    compared = differing = 0
    for path in sorted(Path(args.traces).glob("*.json")):
        match = NAME.fullmatch(path.name)
        if not match or match[3] == "0":
            continue
        plain_path = path.with_name(f"{match[1]}_{match[2]}_k0.json")
        if not plain_path.exists():
            continue
        compared += 1
        plain, spec = tokens(plain_path), tokens(path)
        at = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
        if at is None:
            if len(plain) != len(spec):
                raise SystemExit(f"{path.name}: one output is a prefix of the other")
            continue
        differing += 1
        key = (match[1], match[2], at, plain[at], spec[at])
        if key in seen:
            continue
        seen.add(key)
        ids = tok.apply_chat_template(prompts[(match[1], match[2])], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        context = torch.tensor([[tok.bos_token_id] + list(ids) + plain[:at]]).cuda()
        line = f"{match[1]} {match[2]}: output token {at}, plain {tok.convert_ids_to_tokens([plain[at]])[0]!r}, with drafts {tok.convert_ids_to_tokens([spec[at]])[0]!r}"
        for dtype, model in models.items():
            with torch.no_grad():
                logits = model(context).logits[0, -1].float()
            top = float(logits.max())
            ulp = 2.0 ** (math.floor(math.log2(abs(top))) - 7)
            a, b = float(logits[plain[at]]), float(logits[spec[at]])
            line += f"; {str(dtype).split('.')[-1]}: {a:.4f} and {b:.4f}, top {top:.4f}, {abs(a - b) / ulp:.2f} BF16 steps apart"
        print(line)
    print(f"{compared} traces with drafts compared with their run without, {differing} differ, {len(seen)} different first differences")


if __name__ == "__main__":
    main()
